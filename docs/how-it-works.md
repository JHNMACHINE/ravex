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

### The epoch that was already done

When a checkpoint lands on an epoch boundary, the fast-forward skips every
batch in the next pass, which then yields nothing. That looked harmless and is
not: an empty pass is a signal frameworks act on. HuggingFace `Trainer` reads
"no batches this epoch" as an exhausted dataset and stops training on the spot,
so a resume that happened to land on a boundary ended the run instead of
continuing it — silently, with a clean exit code.

So the dataloader wrapper rolls straight into the next epoch instead. Two
details make that correct rather than merely convenient:

It restarts through the loader's own `__iter__`, not by re-iterating the
sampler. That call draws a worker base seed from the global RNG exactly as the
original run's next epoch did; imitating it by hand would be one draw off, and
every dropout mask after the resume would differ.

It also advances the sampler's epoch by hand, and widens the `set_epoch` shift
to match. A `DistributedSampler` orders its data by epoch number, and the
rollover runs two epochs inside one turn of the user's loop — without the
bump it would replay the epoch it just skipped.

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

That wall time is time the loop is stopped, so the log line for each checkpoint
carries the breakdown rather than only the total — the shape of it being:

```
Checkpoint at step 240 handed off in 0.412s (collect 0.031s, flatten 0.220s, store 0.161s)
```

`collect` is gathering the state, `flatten` the walk through the state tree, and
`store` the shadow copy — plus, if the previous checkpoint's writer has not
drained, however long that took: Moonclip allows one write in flight. Setting
`MOONCLIP_PROFILE=1` separates those two. The `torch_save` backend reports `copy`
and `queue` instead, which is the same split.

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

### Sharded models

FSDP splits every parameter across ranks, and the optimizer's moments with it,
so `state_dict()` on any one rank returns a fragment. Worse, the two cannot be
collected independently: the moments are sharded against the model's flattened
parameters, and gathering them needs both objects at once. Ravex therefore
pairs each sharded model with the optimizers that own its parameters and hands
both to `torch.distributed.checkpoint.state_dict`, which produces a full state
dict with clean parameter names.

The result is a checkpoint that does not remember how many GPUs wrote it. Eight
ranks in, one rank out.

Two things follow, and both are load-bearing:

**Collecting becomes a collective.** The rank gate had to move: every rank
gathers, then only rank 0 writes. Gating before the gather — the obvious
reading of "only rank 0 checkpoints" — deadlocks, because the other ranks sit
in an all-gather waiting for a participant that already returned.

**There is no checkpoint at exit.** Shutdown is exactly where the ranks stop
being in lockstep: user code may have called `destroy_process_group`, or one
rank may reach `atexit` before another. A collective nobody else joins does not
raise, it hangs. So for sharded models the final checkpoint is skipped and the
periodic cadence is what you get. A bounded loss of a few steps beats an
unbounded hang.

### One writer, whatever the environment says

Moonclip supports genuine multi-rank checkpoints, where each rank writes its own
shard through an explicit `create_snapshot` / `save_rank` / `finalize` flow, and
it infers the world size from `RANK` and `WORLD_SIZE` when it is not told
otherwise. Ravex does not use that flow: it gathers sharded state itself and
exactly one rank writes the result. So the backend states `world_size=1` rather
than letting the environment speak for it.

Without that, every distributed run under `torchrun` produced this and nothing
else:

    Multi-rank save requires explicit create_snapshot/save_rank/finalize flow.
    Ravex disabled (checkpoint failed) - training continues unaffected

Which is the fallback behaving exactly as designed — and is also the worst
possible outcome, since the job carries on happily with no checkpoints at all,
on precisely the runs that are expensive enough to be worth checkpointing.

### The step that never happened

Loading sharded optimizer state calls `optimizer.step()` — on purpose, with
zero gradients, to allocate the state tensors before filling them in. PyTorch
says so in as many words:

> `_init_optim_state`: initialize optim states by calling the step() with zero
> grads.

That step goes straight through the post-step hook, which cannot tell it apart
from a real one. Left unguarded, every FSDP resume silently burns one step of
the budget and misnumbers every checkpoint after it — and since each restart
adds another, a job preempted ten times ends ten steps short with checkpoint
names that no longer mean what they say. Nothing errors; the numbers just drift.

So the runtime raises a flag for the duration of a restore, and the step hook
returns immediately while it is set. Anything the loading machinery does to the
optimizer is by definition not training.

This one only shows up on a GPU: FSDP1 refuses to initialise without an
accelerator, and the CPU-only FSDP2 path happened not to hit it.

### Keys

Keys for sharded models are positional (`sharded_0`), not structural
fingerprints. A fingerprint built from parameter shapes would encode the world
size, since each rank only sees its own shard — and a run sharded over eight
GPUs would then fail to recognise itself on four. The parameter names are
stored alongside so a changed architecture is reported rather than surfacing as
a shape error from inside the loader.

### More than one machine

Most of what is above crosses machines untouched. `get_rank()` returns the
global rank, so two nodes of eight are ranks 0-15 rather than 0-7 twice; the
per-rank directories derived from it do not collide; and the collectives that
agree on a step travel the network like any other.

What does not carry over is the assumption that a checkpoint is in one place.

With `per_rank` and local storage every machine writes only its own ranks'
shards, to its own disk, and no machine holds the whole checkpoint. It resumes
only if every machine is handed the same ranks again — which neither `torchrun`
nor SLURM promises — and not at all if a machine is lost. On a six-node bench
inverting the node order was enough: every rank found nothing and the run
started over. The behaviour is safe, since nobody resumes from a checkpoint
they hold half of, but it is total, and it used to be silent.

So the topology is announced at activation rather than discovered at the first
failed resume. Whether the storage is shared is **probed, not inferred from the
path**: every rank drops a uniquely named marker and looks for everyone else's,
because a local disk and an NFS mount are the same `type: local` pointing at a
directory that exists. Three outcomes, and the log says which one you are in —
shared storage, split storage with copies between machines, split storage
without them.

**A bucket is durability, not a way back.** Until 2026-08-19 the remote support
was push-only: `sync_now()` sent local to remote and nothing read the other
way, so the step to resume from was still read from the *local* manifest. On
the bench a node whose disk had been replaced started from scratch with its own
data sitting in the bucket, and took every other rank with it, because agreeing
on a step takes the minimum. Moonclip can now pull a store back when the local
one is empty, which makes a bucket the cheapest answer to both a lost machine
and a reshuffle.

**With neither a bucket nor a shared filesystem**, `replicate_every` is what
stands between you and a lost machine. Each rank collapses its store into one
self-contained full and sends it to a peer picked to land on a different
machine — `(rank + local_world_size) % world_size`, which only holds when the
ranks are spread evenly, so an uneven layout is reported as *not* replicating
instead of being assumed to work. The exchange is point-to-point `isend` /
`irecv`, never a collective: an all-gather would leave every rank holding
`world_size` copies, tens of GiB per process to protect against one loss. It is
all-or-nothing — three copies of four landing is not a restore point, and
recording it as one would be worse than skipping the round. A replica lands
under `replica/` rather than beside the real stores, so discovery cannot
mistake it for a rank's own, and it counts only once a completion marker is
written last.

The guarantee is worth stating exactly: **the loss of any one machine is
survivable, at a cost of at most one replication interval of progress.** Not
"nothing is ever lost".

**Keeping the checkpoints once the run ends is yours.** With no remote
configured they go away with the machines. And for a sharded model there is no
checkpoint at exit at all — see [Sharded models](#sharded-models) — so the most
recent thing worth copying off is the last periodic checkpoint, not something
written on the way out.

## Processes that never train

The backend is not built at activation. It is built the first time something
needs to read or write.

The reason is every process that imports torch inside a project with a
`ravex.yaml` and then never sees a single step: dataloader workers, and any
helper script in the same directory. Building a checkpoint manager eagerly would
mean each of those creating directories and, with S3 or R2 configured, opening
connections on behalf of a process with nothing to save.

The `torchrun` launcher used to be the clearest example — it imports torch to
parse its own arguments, so a run with eight ranks announced nine runtimes. That
one is now recognised at the autoloader and never activates at all. Recognising
it means reading `sys.orig_argv`: under `python -m torch.distributed.run`, runpy
imports `torch.distributed` while resolving which module to run, so torch — and
with it the autoloader — fires before `sys.argv[0]` or `__main__.__spec__` say
anything useful. A worker is never mistaken for it: `LOCAL_RANK` is set, and that
answer comes first.

## When it breaks

Every hook is wrapped: the original call runs first, Ravex's bookkeeping second
and inside a `try`. If the bookkeeping raises, the user's call still returns
its result.

If a checkpoint fails, Ravex disables itself, logs it, and training continues
unaffected. A run that loses its checkpointing is recoverable; a run that dies
on a rented GPU is money burned.
