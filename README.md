# Ravex

Transparent checkpoint and resume for PyTorch training. Your training script
does not change — not one line, not one import.

```bash
pip install ravex
```

Ravex has a small Rust core — the reshard planner — so wheels are built per
interpreter for Linux x86_64. On any other platform pip falls back to the source
distribution, which builds if you have a Rust toolchain. Through 0.0.5 Ravex was
pure Python and installed anywhere; what that bought and what it cost is in the
[CHANGELOG](https://codeberg.org/JHNMACHINE/ravex/src/branch/main/CHANGELOG.md),
along with everything else that changed between versions.

Its default checkpoint engine, [Moonclip](https://codeberg.org/JHNMACHINE/moonclip),
ships wheels for the same platform — so `pip install "ravex[moonclip]"` is a
Linux thing, and where it is unavailable Ravex falls back to `torch_save` on its
own.

Drop a `ravex.yaml` next to your code and run what you always ran:

```bash
python train.py
```

The run now checkpoints itself. If the process dies — spot instance reclaimed,
node rebooted, OOM killer, power cut — running the same command again picks up
where it stopped: same weights, same optimizer moments, same LR schedule, same
RNG state, same position in the dataset.

## Why this is not just `torch.save`

Saving the weights is the easy part. What makes a resumed run *continue* rather
than merely *restart from a good place* is everything around them:

| Restored | Why it matters |
|---|---|
| Model weights and buffers | the obvious part |
| Optimizer state | Adam moments; without them the first steps after resume are wrong |
| LR scheduler | resume at the wrong LR and the loss curve visibly kinks |
| AMP `GradScaler` | its loss scale is tuned state, not a constant |
| RNG: torch, CUDA, Python, NumPy | dropout masks and augmentations replay identically |
| Dataset position | you continue with batch 4001, not batch 1 |

Ravex's test suite asserts the strong form of this: a run killed at step 20 and
resumed produces losses that are **bit-identical**, step by step, to the run
that was never interrupted, and the same final weights.

## How you use it

One decorator on the function that trains:

```python
import ravex

@ravex.train_loop(preemption_handler=True)
def train():
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters())
    for batch in loader:
        ...

train()
```

Inside that function nothing changes. Ravex notices the model, the optimizer,
the LR scheduler, the AMP scaler and the dataloader as they are built, counts
every `optimizer.step()`, and writes a checkpoint on the cadence you configured.
Rerun the script and it picks up where it stopped.

If "on its own" is a guess — two models and only one being trained, an optimizer
built before the function was entered — name it:

```python
ravex.track(model, optimizer)
```

**What the decorator buys, beyond being readable.** It is a *boundary*. Ravex
knows where the loop begins and ends, so the teardown is a `finally` rather than
an `atexit` hook hoping to run, and it holds the function **before the model
exists** — which is what resuming onto a different number of ranks needs and
could never have.

**What it replaced.** Through 0.0.5 `pip install ravex` also put a one-line
`.pth` file in site-packages that ran in *every* Python process in the
environment and attached Ravex to any of them that had a `ravex.yaml` above the
working directory. It is gone. It was genuinely zero-code-changes, and it existed
so a platform could turn on checkpointing for code it did not own — a real thing
to give up, given up on purpose. What it cost was that Ravex had to guess where
the loop began, and the guessing is what limited it: with HuggingFace `Trainer`
and Lightning, dataset and RNG replay could not be made exact, and there was no
defined moment of exit to write a final sharded checkpoint at. Both were filed as
accepted limitations. Neither has to be, now.

## Configuration

`ravex.yaml`, anywhere at or above the working directory:

```yaml
checkpoint_every: 500        # optimizer steps between checkpoints
backend: moonclip            # moonclip | torch_save
storage:
  type: local                # local | s3 | r2
  path: ./checkpoints
keep_last: 5
max_steps: null              # optional hard stop, see below
sharded_checkpoints: gather  # gather | per_rank, for FSDP — see below
```

Every option also reads from `RAVEX_*` environment variables, which win over
the file — so a scheduler can override a config committed to the repository:

```bash
RAVEX_CHECKPOINT_EVERY=100 RAVEX_STORAGE_TYPE=r2 RAVEX_STORAGE_BUCKET=runs python train.py
```

Credentials are never read from the config file. Set `RAVEX_S3_ACCESS_KEY` /
`RAVEX_S3_SECRET_KEY`, or the usual `AWS_*` pair.

Full reference: [docs/configuration.md](docs/configuration.md).

## Backends

**`moonclip`** (default) — the [Moonclip](https://codeberg.org/JHNMACHINE/moonclip)
engine: per-tensor delta tracking, so unchanged weights cost zero I/O; zstd
compression; background writes; direct S3/R2 sync.

**`torch_save`** — one `.pt` file per checkpoint, written on a background
thread. Used automatically when Moonclip is not installed. Correct, just larger
and slower.

## How it works

Ravex patches five things in PyTorch and nothing in your code:

- `nn.Module.__init__` and `.train()` — to notice your models
- `Optimizer.__init__` — to attach a step hook to every optimizer
- `DataLoader.__init__` and `.__iter__` — to track the dataset position and to
  find the one moment where a resume can be applied

The step counter advances once per `optimizer.step()`, so gradient accumulation
needs no special handling. Checkpoints are collected at the *top of an
iteration*, never inside one: mid-iteration the LR scheduler has not stepped
yet, and a checkpoint taken there resumes with a stale learning rate.

Collection runs on the training thread — it has to, to be consistent with the
step that just finished — and copies the state; the write itself happens in the
background. What the loop pays for is the copy, not the I/O.

More detail: [docs/how-it-works.md](docs/how-it-works.md).

## Safety

Ravex is designed to be un-noticeable when it works and harmless when it does
not:

- every hook is wrapped; if one raises, your call still returns normally
- if a checkpoint fails, Ravex disables itself and logs it — training continues
- nothing is ever written to stdout; logs go to `log_file`, or to stderr at
  WARNING and above
- installing the package starts no checkpointing anywhere, and neither does
  importing it. Ravex does nothing at all until a decorated function is called,
  which is a much easier promise to check than the one this used to make: it is
  visible in the source of the thing being checkpointed
- the patches go on at that call and come off when it returns, including when it
  returns by raising. A process that finishes training is a process with an
  unpatched PyTorch in it
- `RAVEX_ENABLED=0` turns it off for a single run without touching the code

## Status and limits

Alpha. Works with plain PyTorch loops, and with anything built on them, since
the hooks are on PyTorch itself.

### With a framework driving the loop

HuggingFace `Trainer` and Lightning are covered by their own tests, and the
result deserves to be stated precisely rather than as "it works":

- **State restoration is exact.** Model, optimizer, LR scheduler and step count
  all come back. With the per-step randomness removed, a killed run resumes
  into a loss sequence identical to the uninterrupted one.
- **Replay is not.** With shuffling and dropout on, the resumed run continues
  correctly from the checkpointed state but sees a different draw. Both
  frameworks iterate the dataloader on their own schedule and consume the
  global RNG around the loop, so the epoch-start snapshot no longer lines up.

Plain loops, DDP and FSDP *are* bit-exact with randomness on. This is a
framework-interaction limit, not a general one, and it costs you a different
shuffle from the resume point onwards — not a wrong model.

Verified: plain loops, gradient accumulation, LR schedulers, AMP loss-scale
state, `num_workers > 0`, DDP, and FSDP. A killed `torchrun` job resumes on
*every* rank with bit-identical losses, sharded or not, and the checkpoint it
leaves behind loads into a plain single-process model afterwards.

### Sharded models

With FSDP each rank holds a slice of every parameter, so `state_dict()` returns
a fragment. Two ways to turn that into a checkpoint, picked with
`sharded_checkpoints`:

**`gather`** (default) rebuilds the whole state on rank 0, which writes it. The
checkpoint is then independent of the topology that produced it — eight GPUs in,
one out — and it does not scale: rank 0 has to hold the entire model and
optimizer in host memory, and it is the rank that then does the writing.

**`per_rank`** has every rank write its own shard into its own store,
`<storage.path>/rank_<n>`. Nothing is gathered, so nothing is bounded by one
rank's memory, and on a 1.48B model collecting the state went from 15.6 s to
1.5 s. What you give up is the resharding: those shards are cut for one topology,
so the checkpoint resumes at the same world size and starts clean at any other.
Needs FSDP2 — under FSDP1 Ravex degrades to `gather` and says so.

Either way, collecting is a **collective**: every rank participates, and there is
**no final checkpoint at exit** for a sharded model. Shutdown is where ranks stop
being in lockstep, and a collective nobody else joins hangs. Losing the last few
steps is bounded; a hang is not. Set `checkpoint_every` accordingly.

Numbers and the FSDP1 details: [docs/configuration.md](docs/configuration.md).

### More than one machine

Ranks, directories and collectives all cross machines unchanged. What does not
is `per_rank` on local storage: each machine writes only its own ranks' shards
to its own disk, so no machine holds a whole checkpoint. It resumes only if
every machine is handed the same ranks again — no launcher promises that — and
not at all if a machine is lost. Ravex probes the storage at activation and
says which case you are in rather than letting you find out at the first resume.

Two ways out, and they are not equivalent:

- **A bucket** (`storage.type: s3`). Checkpoints leave the machines on their
  own, and since 0.0.4 a rank that comes up with an empty disk pulls its store
  back. Before that the remote was push-only — a backup you could not resume
  from.
- **`replicate_every`**, when there is no bucket and no shared filesystem. Each
  rank copies its store to a peer on another machine every N checkpoints.
  Survives losing any one machine, at a cost of at most N checkpoints of
  progress.

Keeping the data after the run ends is yours unless a remote is configured —
and for a sharded model the newest checkpoint is the last periodic one, since
there is none at exit.

Known limits today:

- **`IterableDataset`**: no index sampler exists, so the stream position cannot
  be replayed. Everything else is still restored.
- **Your loop's bounds**: a resumed script runs its own `for epoch in
  range(N)` again from the top; it has no idea 3000 steps already happened. Set
  `max_steps` and Ravex ends the run at the right step regardless of how many
  times the process restarted.

Under AMP, note that an overflowing gradient makes `scaler.step()` skip the
optimizer. Ravex counts optimizer steps, not loop iterations, so a skipped
iteration does not advance the counter — which is the right unit, since nothing
about the model changed, but it does mean the step count and the number of
batches you fed differ.

The GPU paths — AMP with real fp16 overflow, the CUDA RNG, FSDP1, NCCL — are
covered by `integration/test_cuda.py`, which skips without a GPU. They were
last verified on 8× RTX 5060 Ti with torch 2.12/cu130.

## Project layout

Everything under `ravex` is private except `ravex` itself: the public surface
is `train_loop()`, `track()`, a handful of small helpers (`checkpoint`, `flush`,
`step`, `status`, `is_active`, `deactivate`), `__version__` and the `ravex`
command. Every name below that starts with an underscore is free to move.

```
ravex/
├── __init__.py       Public surface, and nothing else
├── _cli.py           ravex status
├── _core.pyi         Types for the compiled Rust core
├── _config.py        Defaults < ravex.yaml < RAVEX_*
├── _patches.py       The five monkey patches on PyTorch
├── _registry.py      What is being trained, held by weakref
├── _runtime.py       One per process; what the patches call into
├── _resume.py        Best-effort restore
├── _backends.py      moonclip | torch_save
├── _sampler.py       Dataset position
├── _frameworks.py    HF Trainer / Lightning / Accelerate detection
├── _dist/            More than one GPU, more than one machine
│   ├── collectives.py    gather vs per_rank; the SIGTERM channel
│   ├── reshard.py        8 shards onto 4 ranks — re-exports the Rust core
│   ├── identity.py       Who wrote this store, as part of which run
│   ├── replication.py    Each rank copies its store to a peer
│   └── elastic.py        Membership changes without a restart
└── _interop/         Checkpoints somebody else wrote
    ├── foreign.py        What is this directory? Layout first, fields second
    ├── zero.py           DeepSpeed ZeRO stages 1–3
    ├── dcp.py            torch.distributed.checkpoint — FSDP, Megatron-core
    ├── convert.py        Into the shape the resume path already consumes
    └── resume.py         When to act on all that, and when to decline
```

The two subpackages are groups, not layers: neither re-exports anything, and
callers import the submodule they want inside the function that wants it. That
used to be a startup-cost rule — the `.pth` ran in every interpreter on the
machine, so an import that did not have to happen was paid for by every
`python -c` on the box. The `.pth` is gone and the discipline is kept for the
weaker reason that still holds: `_dist.collectives` pulls in
`torch.distributed`, and a single-process run should load neither subpackage.

The import graph is a DAG with `_runtime` as its only hub; there are no cycles,
and nothing in `_interop` is imported by anything outside it except `_runtime`.

**And around them**

| Path | |
|---|---|
| `tests/` | 821 unit tests, in-process, no GPU and no container. Named for what they cover: `test_dist_*`, `test_interop_*` |
| `integration/` | What only exists across a real process boundary — a resume from an empty interpreter, `torchrun`, a SIGKILL that runs no `finally`. Linux, in Docker |
| `integration/scripts/` | The training scripts those tests kill and restart |
| `integration/multinode/` | One container per rank, for questions `--nproc_per_node` cannot ask ([README](integration/multinode/README.md)) |
| `integration/two-machines/` | The rented-box harness: two real hosts, real network |
| `docs/` | [configuration.md](docs/configuration.md), [how-it-works.md](docs/how-it-works.md) |
| `src/` | The Rust core: `reshard.rs` is the planner, `python.rs` is the only file that knows an interpreter exists |
| `.forgejo/workflows/` | `checks.yml` on branches; `ci.yml` on main adds the moonclip backend and both integration jobs |

As of 0.0.5 that is about 10.5k lines across 23 modules.

## Development

```bash
pip install -e ".[dev]"
pytest
```

That editable install compiles the Rust core into the source tree, so it needs a
toolchain: `rust-toolchain.toml` names the version and rustup will fetch it. The
half of the engine that has no Python in it has its own tests, and they are the
faster gate — no interpreter, no torch, milliseconds:

```bash
cargo test --no-default-features
```

The unit suite runs in-process. The parts that only exist across a real process
boundary — a resume starting from an empty interpreter, config discovery from
the working directory, `torchrun`, and a SIGKILL that runs no `finally` — live
in `integration/` and need Linux:

```bash
docker build -f integration/Dockerfile -t ravex-integration .
docker run --rm ravex-integration
```

## Licence

Apache 2.0. See [LICENSE](LICENSE).
