# How it works

Ravex has to solve three problems, and most of the design follows from the
third one being much harder than it looks.

1. **Find the training objects** without being told where they are.
2. **Know when a step happened**, whatever the training loop looks like.
3. **Pick a moment where the state is consistent** — and get the RNG right.

## Startup

`ravex enable` writes one line into site-packages:

```
import ravex._bootstrap
```

Python executes `.pth` files at interpreter startup, before any user code. That
module is deliberately tiny: it imports nothing heavy, not even torch.
Importing torch at startup would add seconds to every `python -c` on the
machine and pull CUDA initialisation into processes that never asked for it.

Instead it checks whether this process should activate at all — an explicit
`RAVEX_ENABLED`, or a `ravex.yaml` at or above the working directory — and if
so installs a meta-path finder that waits for `import torch`. Only when torch
actually loads does the runtime get built and the patches installed. A process
that never imports torch pays nothing.

## Finding models

`nn.Module.__init__` is patched, so every module ever constructed is recorded
through a weak reference. That is thousands of objects for a large model, which
is why the registry is keyed by object identity — a linear membership scan
would make model construction quadratic.

Which of those are *models* is decided later, at checkpoint time, in two
filters:

1. **Outermost only.** A module contained in another observed module is already
   covered by its parent's `state_dict()`.
2. **Actually part of the training.** A root qualifies if an optimizer owns at
   least one of its parameters, or if the user explicitly called `.train()` on
   it. The first condition drops the incidental modules libraries build behind
   the scenes; the second catches EMA copies and frozen teachers, which no
   optimizer owns but the user does put in training mode.

### Keys that survive a restart

`id()` is meaningless across processes, so each object gets a *structural
fingerprint*: a digest of its parameters' names, shapes and dtypes. The same
architecture rebuilt in a fresh process lands on the same key and the
checkpoint applies. Change the architecture and the key changes, the state does
not match, and the resume is skipped with a warning instead of exploding.

The digest is `blake2b`, not `hash()`: Python randomises string hashing per
process, so `hash()`-based keys would differ between the run that saved and the
run that resumes — the exact scenario this has to work in.

## Counting steps

`torch.optim.Optimizer.__init__` is the one choke point every optimizer
subclass goes through, so that is where the step hook is attached — per
instance, via `register_step_post_hook`.

Patching `Optimizer.step` on the class would do nothing useful: `Adam`, `SGD`
and every other concrete optimizer override `step` and never call
`super().step()`.

Two consequences fall out for free:

- **Gradient accumulation** needs no detection. `step()` fires once per
  accumulation cycle by construction, so the counter is already in the right
  unit.
- **Multiple optimizers** (a GAN's generator and discriminator) would otherwise
  double-count. Only the first optimizer registered drives the counter, so one
  training iteration stays one step.

## The consistent moment

This is where the subtleties are.

### Not inside the step hook

The obvious place to checkpoint is the optimizer's post-step hook. It is wrong.
A typical loop is:

```python
optimizer.step()      # <- post-step hook fires here
scheduler.step()      # <- has not happened yet
```

State captured in the hook has an LR scheduler that is one step behind, and the
optimizer's `param_groups` still hold the pre-decay learning rate. A run
resumed from it trains with the wrong LR from its very first step. The loss
curve does not crash — it just quietly differs, which is worse.

So the step hook only raises a flag. The actual collection happens at the top
of the next iteration, in a thin wrapper around the dataloader's iterator,
where the previous iteration is complete and the next has not started.

The cost is that a hard kill loses up to one extra iteration. That is the right
trade: a checkpoint that resumes *exactly* beats a checkpoint that is one
iteration fresher and slightly wrong.

### Where the dataset was

`TrackedSampler` wraps the loader's `batch_sampler` (or its `sampler` when
auto-collation is off) and records how far into the epoch the *training loop*
has got. Not how far the sampler has got: with `num_workers > 0` the sampler
runs ahead by the prefetch depth, and the loop's count is the one that
describes where training actually is.

Fast-forward on resume happens at the index level. The skipped batches are
lists of integers produced in the main process — no sample is fetched,
collated, or sent to a worker. Skipping 50k batches costs milliseconds.

For the replayed order to match, the shuffling RNG has to be back where the
epoch started, so the sampler's generator state is captured at `__iter__` time,
not at checkpoint time. And an unseeded `RandomSampler` gets an explicit
generator attached: without one, PyTorch draws a fresh seed from the global RNG
every epoch and no restart can reproduce the order.

What is never attempted is *predicting* where the next epoch will start.
Tempting when the checkpoint lands on the last batch, but wrong:
`RandomSampler` draws an extra permutation when its iterator is exhausted,
which has not happened yet at that point. Ravex always records *(the state the
epoch was drawn from, how many batches were consumed)* and replays. If that
means skipping a whole epoch, the epoch is skipped in microseconds and training
continues with the next one.

### The RNG draw nobody expects

Building a `DataLoader` iterator draws a worker base seed from the **global**
RNG:

```python
self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
```

So restoring the RNG before `iter(dataloader)` leaves the generator one draw
ahead of where the original run was — enough to change every dropout mask from
the first resumed step onwards, and to produce losses that look plausible and
are wrong.

Ravex therefore splits the resume: everything except the global RNG is restored
before the iterator is built, and the RNG state is applied immediately after,
landing on the far side of that draw. The end-to-end test would fail by about
0.5% in the loss without this, which is exactly the kind of discrepancy nobody
would notice in a training log.

## Writing

Collection runs on the training thread. It has to: the state must be consistent
with the step that just finished, and the writer must not read tensors while
the next step mutates them.

What the backend does before returning is copy the state to CPU memory — the
shadow copy. The write itself is background. The loop pays for a copy, not for
I/O.

With Moonclip, the copy is the flatten step: the state tree becomes individual
tensors named `ravex/models/<key>/<param>`, stable across steps, which is what
per-tensor delta tracking keys on. Unchanged weights cost zero I/O on the next
checkpoint.

## Distributed

Rank 0 writes; every rank resumes. That asymmetry matters: a rank that came
back with fresh weights would poison the first all-reduce, and the run would
diverge without ever reporting an error.

DDP and `torch.compile` wrappers are unwrapped before `state_dict()`, so keys
do not pick up a `module.` or `_orig_mod.` prefix that would load fine on the
cluster that wrote them and fail on a single GPU afterwards — which is exactly
when you reach for the checkpoint.

### The epoch counter keeps moving

`DistributedSampler` derives its shuffle entirely from `seed + epoch`, and the
epoch comes from the user's own loop:

```python
for epoch in range(EPOCHS):
    sampler.set_epoch(epoch)
```

A resumed script starts that counter at zero again. Restoring the epoch once
fixes only the first pass; from the second onwards the run would replay epochs
it had already done. So on resume Ravex shifts the sampler's `set_epoch` by the
epoch the checkpoint was taken in. The user keeps counting from zero and the
data keeps moving forward.

FSDP sharded state dicts are not gathered yet.

## Processes that never train

The backend is not built at activation. It is built the first time something
needs to read or write.

The reason shows up as soon as you run `torchrun`: the launcher process imports
torch inside a project that has a `ravex.yaml`, so Ravex activates there too —
and then never sees a single step. Same for dataloader workers, and for any
helper script in the same directory. Building a checkpoint manager eagerly
would mean each of those creating directories and, with S3 or R2 configured,
opening connections on behalf of a process with nothing to save.

## When it breaks

Every hook is wrapped: the original call runs first, Ravex's bookkeeping second
and inside a `try`. If the bookkeeping raises, the user's call still returns
its result.

If a checkpoint fails, Ravex disables itself, logs it, and training continues
unaffected. A run that loses its checkpointing is recoverable; a run that dies
on a rented GPU is money burned.
