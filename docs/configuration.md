# Configuration

Ravex resolves its configuration from three places. Later ones win:

1. built-in defaults
2. `ravex.yaml` — at `RAVEX_CONFIG`, or the first one found walking up from the
   working directory
3. `RAVEX_*` environment variables

Environment variables come last on purpose: a platform or a CI job has to be
able to override a config file committed to the user's repository.

Check what a given directory resolves to:

```bash
ravex status
```

## Options

| Key | Env var | Default | What it does |
|---|---|---|---|
| `enabled` | `RAVEX_ENABLED` | `true` | Master switch. `RAVEX_ENABLED=0` disables Ravex for one run. |
| `checkpoint_every` | `RAVEX_CHECKPOINT_EVERY` | `500` | Optimizer steps between checkpoints. Not micro-batches: with gradient accumulation, one accumulation cycle is one step. |
| `checkpoint_on_exit` | `RAVEX_CHECKPOINT_ON_EXIT` | `true` | Take a final checkpoint when the process exits normally or on SIGTERM. |
| `resume` | `RAVEX_RESUME` | `true` | Look for an existing checkpoint at startup. Set false to always start clean. |
| `max_steps` | `RAVEX_MAX_STEPS` | `null` | Hard stop, in optimizer steps. See [Step budgets](#step-budgets). |
| `backend` | `RAVEX_BACKEND` | `moonclip` | `moonclip` or `torch_save`. Falls back to `torch_save` if Moonclip is missing. |
| `delta` | `RAVEX_DELTA` | `true` | Moonclip only: store deltas against the previous snapshot. |
| `keep_base_in_memory` | `RAVEX_KEEP_BASE_IN_MEMORY` | `true` | Moonclip only: keep the last full snapshot's bytes resident so the next delta does not have to read them back. Costs a copy of the saved state. See [The write path](#the-write-path). |
| `async_save` | `RAVEX_ASYNC_SAVE` | `true` | Moonclip only: write in the background. Off, the loop stops until the checkpoint is durable. Diagnostic; see [The write path](#the-write-path). |
| `compression` | `RAVEX_COMPRESSION` | `zstd` | `zstd` or `none`. |
| `compression_level` | `RAVEX_COMPRESSION_LEVEL` | `3` | zstd level. |
| `keep_last` | `RAVEX_KEEP_LAST` | `5` | Checkpoints to retain. Older ones are deleted. |
| `sharded_checkpoints` | `RAVEX_SHARDED_CHECKPOINTS` | `gather` | How FSDP state is written: `gather` or `per_rank`. See [Sharded models](#sharded-models). |
| `reshard_on_resume` | `RAVEX_RESHARD_ON_RESUME` | `false` | Resume a `per_rank` checkpoint at a different world size, rebuilding each shard from the old ones. See [Resuming onto a different number of ranks](#resuming-onto-a-different-number-of-ranks). |
| `replicate_every` | `RAVEX_REPLICATE_EVERY` | `10` | Checkpoints between copies of each rank's store to a peer on another machine. Only ever used when the storage turns out to be neither remote nor shared. `0` turns it off. See [More than one machine](#more-than-one-machine). |
| `track_dataloaders` | `RAVEX_TRACK_DATALOADERS` | `true` | Track and restore the dataset position. |
| `track_rng` | `RAVEX_TRACK_RNG` | `true` | Save and restore torch / CUDA / Python / NumPy RNG state. |
| `handle_sigterm` | `RAVEX_HANDLE_SIGTERM` | `true` | Checkpoint on SIGTERM — the signal a preempted spot instance receives. Only installed if nothing else has claimed the signal. |
| `fallback_on_error` | `RAVEX_FALLBACK_ON_ERROR` | `true` | On an unexpected error, log it and let training continue; the checkpoint is retried at the next one. Set `false` to raise instead. |
| `log_file` | `RAVEX_LOG_FILE` | `null` | Log destination. Unset means stderr, WARNING and above only. |
| `log_level` | `RAVEX_LOG_LEVEL` | `INFO` | |
| `run_id` | `RAVEX_RUN_ID` | `null` | Recorded in checkpoint metadata; also used as the storage prefix when none is set. |
| `frameworks.auto_detect` | — | `true` | Whether to identify the training framework in use. File only: there is no environment variable for it. |

### Sharded models

With FSDP each rank holds a slice of every parameter and of the optimizer
moments beside it. There are two ways to turn that into a checkpoint, and they
trade against each other.

**`gather`** (default) collects the whole unsharded state on rank 0, which
writes it. The checkpoint is then independent of the topology that produced
it — eight GPUs in, one out — and it does not scale.

**`per_rank`** has every rank write its own shard into its own store,
`<storage.path>/rank_<n>` (and the same suffix on `storage.prefix` for a remote
store). Nothing is gathered, so nothing is bounded by one rank's memory, and
each rank keeps its own background writer, delta chain and retention.

Measured on 8× RTX 5060 Ti, FSDP2, a 1.48B model with Adam — 16.5 GiB of state
(5.5 model + 11 optimizer):

| collecting the state | wall time | held | peak host RSS | still on device |
|---|---|---|---|---|
| `gather` | 15.6 s | 16.5 GiB on rank 0 | 18.1 GiB on rank 0 | 0/44 |
| `per_rank` | 1.5 s | 2.1 GiB per rank | 6.1 GiB, every rank alike | 0/44 |
| `per_rank`, shards left on device | 4 ms | 2.1 GiB per rank | 6.1 GiB | 44/44 |

Three things in that table are worth naming. The gather is **10× slower** than
taking the same state per rank, and it is slower on the rank that then has to
do the writing. Its 18.1 GiB of resident memory is the whole state on one
process; per-rank, no rank grows at all past the model it already had. And the
last row is not really a copy: `to_local()` is a view, so 4 ms is bookkeeping
and nothing else — the device→host copy has not happened yet, which is the
point. Under `gather` torch does that copy itself, into pageable memory, before
Moonclip is handed anything; left on the device it goes through Moonclip's
pinned staging instead (9.4× on the transfer, measured separately).

Ravex uses the middle row today: shards are taken with `cpu_offload=True`. The
bottom row is what the plumbing allows, not what it does.

What you give up is the resharding — unless you ask for it back. Per-rank
shards are cut for one topology and compose into nothing on another, so by
default the checkpoint only resumes at the same world size with the same
sharding, and resuming at a different one starts the run clean on *every* rank
(a resume half the ranks complete leaves the others in collectives nobody
joins). `reshard_on_resume` rebuilds the shards instead; see
[Resuming onto a different number of ranks](#resuming-onto-a-different-number-of-ranks).

Two further consequences worth knowing:

- Ranks write independently, so a kill can leave rank 3 holding step 8 and rank
  5 only step 6. On resume the ranks agree on the newest step *all* of them
  have and load that one.
- `per_rank` needs FSDP2. FSDP1 falls back to `gather` with a warning —
  including with `use_orig_params=True`, which is worth stating because it
  sounds like it should be enough: measured on torch 2.12, FSDP1 leaves the
  parameters as plain `Parameter`s and hands back a sharded state dict of
  `ShardedTensor`, which carries no mesh and no placements and so cannot be
  put back shard by shard.

### Resuming onto a different number of ranks

`reshard_on_resume: true` lets a `per_rank` checkpoint be resumed at a world
size it was not written at. Each rank rebuilds its own shard out of the old
ones: the shards are measured rather than recomputed, so the arithmetic never
depends on reproducing how torch chunks a tensor, and no rank ever holds the
whole tensor.

```yaml
sharded_checkpoints: per_rank
reshard_on_resume: true
```

**Off by default, and not out of caution about the arithmetic.** A resume that
reshards silently is a resume that silently succeeds when the launcher started
three ranks where the job wants four — the run continues, the loss looks
plausible, and nothing says the world shrank. So the mismatch is *always*
detected and logged; only acting on it is opt-in.

Two preconditions, both refused loudly rather than worked around:

- **A 1-D mesh** — FSDP, `Shard` and `Replicate`. A 2-D mesh (FSDP crossed with
  tensor parallel) makes the plan a cartesian problem rather than an interval
  one and is not attempted.
- **Every old rank's store readable from here**, as itself or as a complete
  peer copy (see [`replicate_every`](#more-than-one-machine)). Every old shard
  is needed, including the ones this rank reads nothing from: where a shard
  starts is the running sum of all the lengths before it, so one missing store
  is an unknown alignment for everybody. Carrying on without it would produce
  tensors with a band of uninitialised rows — every shard valid, the model
  wrong, and nothing downstream able to notice.

Local storage only, so far. On remote storage the mismatch is reported and the
run starts from scratch.

**What a resharded resume does not promise.** It continues the *model*, not the
run, and the two differences are worth knowing before you rely on it:

- **The data order.** The sampler partitions the epoch by world size, so at a
  different world size each rank walks different samples in a different order.
  The position is rescaled to preserve the total amount of data consumed, and
  every sample is still seen once per epoch — but it is not bit-identical
  continuation and it cannot be.
- **Per-rank RNG.** There were N generator states and there are now M ranks;
  there is no correct mapping. The saved states are not restored, and random
  draws continue from whatever seeding your script did. A log line says so.

### The write path

Two options change what a checkpoint costs, and both default to the fast side.
The numbers below are from a 1B-parameter model with Adam on 8x RTX 5060 Ti —
they will not transfer exactly, but the shape of the trade will.

**`keep_base_in_memory`** holds the last full snapshot's bytes so Moonclip can
compute the next delta against them. That is one extra copy of the saved state
resident for the life of the run: about +11 GiB at that size. Turning it off
does **not** make collection cheaper — `collect` measures identical — it moves
the cost to `store`, which goes from 1.0 s to 20-23 s because the base is read
back from storage for every delta. It is an option for a run with memory as the
binding constraint, paid for in I/O. It is not an optimisation.

**`async_save`** off makes the training loop wait for the checkpoint to be
durable, which took the average total stall from ~10 s to ~27 s per checkpoint
on that same configuration. It exists to make timings attributable while
diagnosing something, not to make a run safer. The background write is already
durable before the next one starts.

### Reading the handoff log

At `INFO` each checkpoint logs its phases:

```
Checkpoint at step 200 handed off in 13.6s (drain 11.1s, collect 1.4s, store 1.0s)
```

**`drain` is not a cost of checkpointing.** It is the wait for the accelerator
queue to empty, and CUDA is asynchronous: at a wide `checkpoint_every` the
training steps since the last checkpoint have been queueing work that comes due
right there, because that is the first point anything asks for it. At
`checkpoint_every=20` it was 11.1 s of a 13.6 s handoff — the largest entry, and
none of it caused by the checkpoint. Reading it as checkpoint overhead leads to
the wrong conclusion; that mistake is what
[GPU-54](https://linear.app/gpuzero/issue/GPU-54) cost.

The phases that *are* the handoff are `collect` (building the state dict, a
collective on a sharded model), `store` (the shadow copy the backend takes so
the loop can carry on mutating weights), and `backpressure` (waiting for the
*previous* checkpoint's writer, which allows one write in flight).

`store` and `backpressure` are reported apart because they answer to different
things: `store` is memory bandwidth and grows with the model, `backpressure`
grows with the cadence and with the speed of the storage. A large
`backpressure` means checkpoint less often or write somewhere faster; a large
`store` means the state is big, and no cadence will change it.

### Storage

| Key | Env var | Default |
|---|---|---|
| `storage.type` | `RAVEX_STORAGE_TYPE` | `local` |
| `storage.path` | `RAVEX_STORAGE_PATH` | `./checkpoints` |
| `storage.bucket` | `RAVEX_STORAGE_BUCKET` | `null` |
| `storage.prefix` | `RAVEX_STORAGE_PREFIX` | `""` |
| `storage.endpoint` | `RAVEX_STORAGE_ENDPOINT` | `null` |
| `storage.region` | `RAVEX_STORAGE_REGION` | `us-east-1` |
| `storage.path_style` | `RAVEX_STORAGE_PATH_STYLE` | `false` |

`type: s3` and `type: r2` are the same S3 protocol; R2 just needs an
`endpoint`. With a remote store, `storage.path` stays in use as the local
staging directory — checkpoints land on local disk first and sync from there,
so a step is never blocked on the network.

A remote type without a bucket falls back to local storage with a warning
rather than failing the run.

### Credentials

Never in `ravex.yaml` — that file lives in the user's repository. Ravex reads,
in order:

1. `RAVEX_S3_ACCESS_KEY` / `RAVEX_S3_SECRET_KEY`
2. `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`

### More than one machine

A job spanning several machines with `sharded_checkpoints: per_rank` and local
storage splits one checkpoint across disks that cannot see each other. Ravex
detects this at activation — by probing the storage, not by reading the path —
and says so in the log.

`replicate_every` is the answer when there is no bucket and no shared
filesystem: every this many checkpoints, each rank sends a copy of its store to
a peer on another machine.

```yaml
sharded_checkpoints: per_rank
replicate_every: 10
```

The interval is what keeps the copy from becoming backpressure on the training
loop — average bandwidth is one store per rank divided by it — and what it buys
is bounded in the same breath: losing a machine costs at most that many
checkpoints of progress.

Left at its default it costs nothing on a job that does not need it. With a
bucket or a shared filesystem the copies stay off, because there the checkpoint
is already reachable from anywhere and the bandwidth would buy nothing. An
uneven spread of ranks over machines also turns them off, and says so, since no
peer can then be shown to be on a different machine.

`storage.type: s3` covers the same ground more cheaply when you have a bucket:
checkpoints leave the machines on their own, and a rank that comes up with an
empty disk pulls its store back. See
[More than one machine](how-it-works.md#more-than-one-machine) for what each
one actually guarantees.

## Step budgets

A resumed script runs its own loop from the top. `for epoch in range(10)` has
no idea that 7 epochs already happened in a previous process, so it does 10
more — from the right state, but well past the intended budget.

`max_steps` fixes this from Ravex's side. Once the counter reaches it, the
dataloader stops handing out batches: the epoch in progress ends there, any
remaining loop iterations fall through instantly, and the script exits on its
own terms — Ravex never raises anything into user code.

```yaml
max_steps: 50000
```

Total optimizer steps across every restart is then exactly 50000, not "50000
rounded up to the end of whatever epoch that fell in".

The unit is optimizer steps, which under AMP is not the same as loop
iterations: an overflowing gradient makes `scaler.step()` skip the optimizer,
and a skipped step does not count because nothing about the model changed.

## Example

```yaml
# ravex.yaml
checkpoint_every: 250
max_steps: 50000
backend: moonclip
keep_last: 3

storage:
  type: r2
  bucket: my-training-runs
  prefix: gpt2-wikitext
  endpoint: https://<account>.r2.cloudflarestorage.com
  path: /workspace/.ravex-cache

log_file: /var/log/ravex.log
```
