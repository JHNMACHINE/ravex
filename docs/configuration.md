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
| `save_dtype` | `RAVEX_SAVE_DTYPE` | `null` | Moonclip only: what precision each part of the checkpoint is stored at. A dtype for everything, or a mapping of component to dtype. See [Precision per component](#precision-per-component). |
| `compression_level` | `RAVEX_COMPRESSION_LEVEL` | `3` | zstd level. |
| `keep_last` | `RAVEX_KEEP_LAST` | `5` | Checkpoints to retain. Older ones are deleted. |
| `sharded_checkpoints` | `RAVEX_SHARDED_CHECKPOINTS` | `gather` | How FSDP state is written: `gather` or `per_rank`. See [Sharded models](#sharded-models). |
| `reshard_on_resume` | `RAVEX_RESHARD_ON_RESUME` | `false` | Resume a `per_rank` checkpoint at a different world size, rebuilding each shard from the old ones. See [Resuming onto a different number of ranks](#resuming-onto-a-different-number-of-ranks). |
| `convert_foreign` | `RAVEX_CONVERT_FOREIGN` | `false` | Resume from a checkpoint another framework wrote — DeepSpeed ZeRO, or a torch distributed checkpoint (which is what Megatron-core writes). See [Resuming from another framework's checkpoint](#resuming-from-another-frameworks-checkpoint). |
| `replicate_every` | `RAVEX_REPLICATE_EVERY` | `10` | Checkpoints between copies of each rank's store to a peer on another machine. Only ever used when the storage turns out to be neither remote nor shared. `0` turns it off. See [More than one machine](#more-than-one-machine). |
| `track_dataloaders` | `RAVEX_TRACK_DATALOADERS` | `true` | Track and restore the dataset position. |
| `track_rng` | `RAVEX_TRACK_RNG` | `true` | Save and restore torch / CUDA / Python / NumPy RNG state. |
| `handle_sigterm` | `RAVEX_HANDLE_SIGTERM` | `true` | Checkpoint on SIGTERM — the signal a preempted spot instance receives. Only installed if nothing else has claimed the signal. |
| `emergency_coordination` | `RAVEX_EMERGENCY_COORDINATION` | `true` | For sharded models split across more than one machine only: wait for every rank to notice a SIGTERM before attempting the collective save FSDP checkpointing needs. No effect on a replicated (DDP) job or a single machine. See [Emergency checkpoint on preemption](#emergency-checkpoint-on-preemption-sharded-models). |
| `emergency_check_every` | `RAVEX_EMERGENCY_CHECK_EVERY` | `1` | Optimizer steps between checks for a SIGTERM on any rank. Only spent when `emergency_coordination` is active for this run (sharded, multi-machine). |
| `emergency_timeout` | `RAVEX_EMERGENCY_TIMEOUT` | `20` | Seconds before the SIGTERM-detection channel gives up, if the rank that raised it disappears before the others get there. Isolated from the main process group's own timeout — see the section below. |
| `emergency_transport` | `RAVEX_EMERGENCY_TRANSPORT` | `auto` | How the ranks learn one of them was preempted: `store` announces the step everybody saves at on the rendezvous store, `collectives` asks with an `all_reduce` every step, `auto` takes the store where there is one. See [What a step of a run nobody preempts pays for this](#what-a-step-of-a-run-nobody-preempts-pays-for-this). |
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

### Resuming from another framework's checkpoint

`convert_foreign: true` lets a run start from a checkpoint Ravex did not write:
a DeepSpeed ZeRO directory, or a `torch.distributed.checkpoint` one — which is
also what `megatron.core.dist_checkpointing` writes, so a Megatron checkpoint
is one of these.

```yaml
convert_foreign: true
storage:
  type: local
  path: /checkpoints/from-the-other-team
```

Nothing else changes, and the training script does not change at all. At resume
Ravex looks at the storage path, recognises what is in it, rebuilds the whole
unsharded tensors, matches their names against the live model, and hands the
result to the same restore path a gathered Ravex checkpoint takes — including
scattering onto FSDP shards if this run has them.

**Off by default, and this is the stronger of the two opt-ins here.** A foreign
checkpoint sitting where Ravex's own store belongs usually means a path points
somewhere unintended. Converting it quietly would turn that into a run trained
on someone else's weights with nothing in the log to say so. The detection is
unconditional and always reported; only acting on it is a setting.

**A DeepSpeed run resuming a DeepSpeed checkpoint is declined.** The engine
loads its own checkpoints; doing it twice by two routes leaves the optimizer
disagreeing with itself. Ravex says so and stands aside.

What comes across and what does not:

- **Weights and optimizer moments**, matched by parameter name. Verified
  bit-exact against DeepSpeed's own `zero_to_fp32` across ZeRO stages 1, 2 and
  3 at several world sizes.
- **Not the step count's meaning, the data order, or the RNG.** A foreign
  checkpoint carries no Ravex sampler position and no per-rank generator state.
  The model continues; the run does not.
- **Not hyperparameters the receiving optimizer lacks.** DeepSpeed's Adam
  records `bias_correction` and torch's does not; it is dropped and named in
  the log rather than translated into something with no correct value.
- **Not an optimizer state keyed by position** onto a sharded model. A plain
  `optimizer.state_dict()` inside a distributed checkpoint numbers its entries
  by the writing run's parameter order; a sharded restore matches by name. The
  weights convert, the moments do not, and the log says which.

Two things worth knowing before pointing `storage.path` at a checkpoint you
did not write:

- **Ravex will write its own checkpoints there too.** After converting, the
  next save lands in the same directory in Ravex's layout, so the directory
  then holds two formats. Point `storage.path` at a copy, or at a fresh
  directory once the conversion has happened.
- **One model.** Converting needs one model to convert into; a run with
  several sharded groups is refused rather than guessed at.

### Elastic training: a cluster that changes size while it runs

A job whose nodes come and go needs no Ravex API of its own. `torchrun` already
has one — its elastic agent negotiates membership between the nodes and
restarts the workers at the new world size — and what Ravex has to do on the
other side of that restart is exactly what it does after any restart: resume.
The resharding above is what makes the resumed world size allowed to differ.

Five things have to be true together, and four of them are the launcher's:

```sh
torchrun --nnodes=1:8 --nproc-per-node=8 \
         --rdzv-backend=c10d --rdzv-endpoint="$HEAD:29500" --rdzv-id=myjob \
         --max-restarts=3 \
         train.py
```

```yaml
sharded_checkpoints: per_rank
reshard_on_resume: true      # the world changing is the point here
storage:
  type: s3                   # or any path every node can see
  bucket: my-checkpoints
```

And in your script, a collective timeout you chose:

```python
dist.init_process_group("nccl", timeout=timedelta(minutes=2))
```

**That last line is not a detail.** The default is **1800 seconds**. When a node
disappears, the survivors do not fail — they block in the collective they were
in, and `torchrun`'s agent restarts a worker group on *failure*, so until that
timeout expires there is nothing for it to react to. A cluster that lost a node
looks perfectly healthy for half an hour. With a timeout in the low minutes the
survivors fail fast, the agent re-rendezvouses, and the run continues.

**The storage has to be shared or remote.** With a local path per machine, each
survivor can see only the shard it wrote itself, and the resharding refuses —
correctly, since rebuilding from the shards that happen to be present is the
uninitialised-rows failure above. Measured on two rented machines on
2026-08-31: moving shards between machines runs at the speed of the link
between them, while the same bytes to object storage went up 5x and came back
down 13x faster. Shared storage is not the fallback here, it is the answer.

Verified end to end in `integration/elastic/probe.sh`, which runs several
`torchrun` agents in one container, kills one, and checks what the survivors
do. On a 3-node job losing a node, the agents re-rendezvoused in about 40
seconds and the run resumed from the last checkpoint at world size 2.

What you give up is what any resharded resume gives up — the data order and the
per-rank RNG, both described above. What you do not give up is the model, the
optimizer moments, or the step count.

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

### Precision per component

`save_dtype` decides what precision each part of a checkpoint is stored at. It
is off by default, and it is the single largest saving available on a run that
checkpoints often.

```yaml
save_dtype:
  optimizer: bf16
```

Keys are component names — `model`, `optimizer`, `scheduler`, `scaler`,
`dataloader` — or raw Moonclip globs over the tensor name for anything they do
not cover. Values are `none`, `bf16`, `fp16`, `fp32`, `fp64`, `fp8`
(= `fp8_e4m3`) or `fp8_e5m2`. A bare value applies to everything:

```yaml
save_dtype: bf16
```

**The first matching rule wins**, in the order written, which is how an
exception is expressed:

```yaml
save_dtype:
  model: none      # the weights, untouched
  "*": bf16        # everything else
```

Written the other way round the catch-all comes first and the exception never
applies.

From the environment, `RAVEX_SAVE_DTYPE=bf16` for the bare form and
`RAVEX_SAVE_DTYPE=model:none,optimizer:bf16` for rules, ordered left to right.

#### Why it is worth setting

Measured on a 1.5B model under FSDP2, 8x RTX 5060 Ti, per rank per checkpoint:

| component | share of the state | what the delta saves |
|---|---|---|
| model weights | ~1/3 | **-70%** |
| `exp_avg` | ~1/3 | -1.1% |
| `exp_avg_sq` | ~1/3 | -4.0% |

The optimizer moments are two thirds of the state and **85% of the bytes
actually written**, because they are the part deltas cannot compress: with
β₁ = 0.9 a tenth of each value is replaced by fresh gradient every step, which
moves nearly every mantissa bit, so the XOR between two steps is noise.

They are also the part that tolerates the least precision. `exp_avg_sq` reaches
Adam through `sqrt(v)`, which halves the relative error — the same reason
8-bit optimizers are ordinary practice rather than an experiment. The weights
are the model and are left alone.

So `{optimizer: bf16}` halves 85% of the volume and changes nothing about the
model. It is the best ratio of saving to risk in the whole write path.

#### What it does not do

Integer, boolean and complex tensors are stored unchanged whatever this says —
step counters, causal masks and indices travel through, they are never cast.
Casting weights to an integer type would be quantization, which needs a scale
and a zero-point a checkpoint entry has nowhere to keep, so those names are
refused rather than accepted into something that produces wrong numbers
quietly.

`fp64` is a valid target but only ever widens: it recovers no precision the
source did not have and writes twice the bytes. It is there for reference runs
that want one width throughout.

The float8 targets quantize against a per-tensor scale. They are a quarter the
size of fp32 and keep four significant bits — a few percent of relative error
on every element, twenty times what bf16 costs. Reasonable for an archived copy,
poor for a checkpoint a run will resume from, and optimizer moments in
particular do not survive it.

#### Why components rather than patterns

An optimizer is written in two places. Unsharded, its state goes under
`ravex/optimizers/…`; under FSDP, the same optimizer goes under
`ravex/sharded/<key>/optimizer/…`, because sharded state has its own
collection path. A pattern written by hand as `ravex/optimizers/*` is correct
on a single GPU and silently casts **nothing** on the sharded run the setting
was chosen for. Naming the component covers both.

Raw patterns still work for anything the component names do not reach, and they
are passed through untouched.

#### Turning it on mid-run

It changes the numbers a resumed run gets back, so it stays off across an
upgrade and is never enabled for you. Turning it on part-way through a run is
safe — a checkpoint records the original dtype of every tensor it cast and
restores it on load — but the steps written before and after are stored at
different precision, and only the later ones are smaller.

An unusable value never stops a run: it is dropped, the run continues storing
that component unchanged, and the reason is logged once at startup along with
every other configuration problem.

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

On a sharded checkpoint with more than one rank, a `skew` phase can appear
before it:

```
Checkpoint at step 200 handed off in 141.2s (skew 138.6s, drain 0.1s, collect 1.4s, store 1.0s)
```

**`skew` *is* a cost, and it is usually the checkpoint's own.** A barrier runs
immediately before the accelerator drain, so what it absorbs — a rank that
reaches the checkpoint before its peers, and waits there — is measured apart
from the device queue. Splitting them mattered because they look identical
from outside: two machines running the same job, one flat at ~35 s of `drain`
and the other oscillating between ~104 s and ~139 s, with the *other* phases of
the same checkpoints identical on both. Compute does not explain a three- to
four-fold difference between otherwise identical ranks; a peer that is late
because it is itself checkpointing does. Measured on two machines, the same
five checkpoints: `skew 34.6s / drain 0.000s` on one rank and `skew 103.8s /
drain 0.000s` on the other — all of what a single `drain` number would have
hidden. See [GPU-98](https://linear.app/gpuzero/issue/GPU-98).

The phases that *are* the handoff are `skew` (waiting for a peer at the
barrier), `collect` (building the state dict, a collective on a sharded
model), `store` (the shadow copy the backend takes so the loop can carry on
mutating weights), and `backpressure` (waiting for the *previous* checkpoint's
writer, which allows one write in flight).

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

### Emergency checkpoint on preemption (sharded models)

**Read the numbers before relying on this.** A preempted spot instance gets
SIGTERM roughly 10s before it is killed. The local handoff alone — copying
state off the GPU and handing it to the backend, no network involved — has
been measured at 10.6s on 8× RTX 5060 Ti (see the CHANGELOG). That is already
the whole budget, before this mechanism's own cost. **This is a best-effort
attempt that will often not complete, not a guarantee that spot training under
FSDP is reliable.** For a plain-replicated (DDP) job the existing SIGTERM
handling already writes a complete checkpoint reliably, with nothing in this
section relevant — this only concerns `sharded_checkpoints: per_rank` split
across more than one machine.

The reason it needs anything beyond `handle_sigterm` at all: extracting even
one rank's own shard goes through PyTorch's own checkpoint machinery
(`torch.distributed.checkpoint.state_dict.get_state_dict`), which is
collective even when nothing is being gathered across ranks. A lone rank
cannot produce a writable shard by itself. So on SIGTERM, the affected rank
raises a flag; every rank checks for that flag at the same cadence
(`emergency_check_every` steps); if any rank has it set, every rank enters the
same save together. If the flag never reaches every rank in time, or the
detection channel itself fails, nothing happens beyond what already happens
today — the last periodic checkpoint stands, same as if this were switched
off.

The detection channel runs on its own process group, separate from the one
carrying gradient synchronization, with its own short timeout
(`emergency_timeout`). That isolation is deliberate: this project's own
two-machine testing has already shown the network between rented boxes to be
flaky enough that shortening the *main* group's timeout would risk killing
otherwise-healthy runs on an ordinary transient blip. A stuck detection round
fails on its own short timeout without ever touching that.

Only the rank that actually received SIGTERM terminates afterward. Every
other rank that entered the save together goes back to ordinary training,
exactly as after any periodic checkpoint — a false alarm costs one
out-of-cadence checkpoint, not a stopped job.

### What a step of a run nobody preempts pays for this

The channel used to ask its question with a collective: an `all_reduce(MAX)`
of one int32 on the isolated group, once per `emergency_check_every` steps.
Perfect agreement, and a collective on every step of a run that will almost
certainly never be preempted at all. Measured on loopback: **376 µs at 4
ranks, 773 at 8** — and 5.8 ms on every rank when one of them is 5 ms late,
because a gather waits for the slowest whatever carries it.

`emergency_transport` (`auto` by default, `store` or `collectives` to pin it)
moves the common case off that collective. The preempted rank writes one key
on the rendezvous store, and the key does not say *whether* — it says **which
step everybody saves at**:

- an ordinary step is now `check` on a key that is not there: **83 µs at 4
  ranks, 162 at 8**, and 92 µs rather than 5.8 ms with a straggler, since
  the absence of a key is answered by the store alone and no rank waits for
  another;
- the collective has not gone away, it has moved to the announced step, where
  it is what proves every rank arrived. A rank that never read the alarm never
  posts it, the others run out of `emergency_timeout`, and nobody saves. That
  matters: without the proof, a rank that missed the alarm would leave the
  others inside a *sharded* collective on the main process group, whose
  timeout is thirty minutes of billed idle GPUs;
- a rank that reads the alarm after the announced step has missed it and does
  not reschedule. Announcing a second round would put the ranks back into
  disagreement, which is the one thing this is for. The fallback is a job with
  the channel off: the last periodic checkpoint stands.

**The announced step is a step number, so it means the same thing on every
rank only where a collective every step keeps the counters together.** That is
DDP and FSDP, which is also the only shape where this channel is on. It is not
true of nodes training independently between outer rounds — see
`outer_loop` below, and GPU-125.

`collectives` is the road back, and it is not deprecated: a job that would
rather pay per step than have its preemption path depend on a key-value store
asks for it by name.

## Training across the internet

Nodes on different continents cannot synchronise gradients. The link between
two rented boxes carries about 7 MB/s, and an all-reduce of a 1B model's
gradients is 3 GB per node — **429 seconds per step**. So instead of
communicating every step, each node trains locally for `outer_inner_steps` and
then the nodes exchange the *difference* between the parameters they started
the round with and the ones they hold now. An outer optimizer takes one step on
the average. The bytes per round are the same as one gradient all-reduce; they
are paid once per H steps.

```yaml
outer_loop: true
outer_inner_steps: 500
outer_round_seconds: 0       # or a wall clock, see below
outer_lr: 0.7
outer_momentum: 0.9
outer_combine: mean          # mean | normalized | step_weighted
outer_deadline: 900
outer_save_dtype: null       # bf16 takes a fifth off a round
outer_root: null             # defaults to <storage.path>/rounds
```

**It never turns itself on.** Everything else in Ravex activates by itself
because the worst it does is write a checkpoint. This changes what the run
*trains* — the weights get averaged with other machines'. `outer_loop` is
`false` until somebody writes otherwise.

**A node dying does not stop the round.** Reports are pulled over Ravex's own
sockets, each with its own deadline, and the round closes over whoever
answered. There is no collective to hang in. `outer_deadline` is that deadline:
running out of it is the answer, not a failure. It bounds a transfer that is
merely *slow* as well as one that never starts: a fetch cut halfway leaves the
round closed over fewer nodes and costs a resend on the next one, not the run.

**What a round costs, measured.** Every round logs where its seconds went —
network, of which the part spent waiting for a peer to reach the round; the
delta; the publish, of which the part spent waiting for a peer's fetch; and the
outer step. On a link throttled to 7 MB/s with 200 ms of round trip, a 15.8 MB
report takes 2.97 s to move, against the 2.26 s the bytes alone are worth: the
transport gets about **76% of the link**, and the rest is decode and store.
Extrapolated, a 1B model's bf16 delta is roughly **375 s per round** — set
`outer_deadline` above that, and `outer_inner_steps` high enough that it is a
small share of the round. `bench/round_link_cost.py` runs that measurement on
any machine, with no privileges and no second box.

**`outer_round_seconds` is what makes nodes of different speeds work.** With a
step count, every node does the same work and the slowest sets the pace. With a
clock, every node stops at the same moment having done as many steps as it
could, and the step counts differing becomes the normal case. Set both and
whichever comes first closes the round.

**`outer_combine` was decided by measurement.** A delta is already proportional
to the work behind it, so weighting *again* by step count counts a fast node
twice. On nodes drawing from the same distribution that is invisible;
give each node its own data and `step_weighted` costs **45% more loss** than
the default. Leave it on `mean` unless you have measured otherwise.

**A loop with no `DataLoader` has to hand over the boundary.** A round is
closed at the top of a training iteration, never inside `optimizer.step()` —
closing it there would write the averaged parameters into the model underneath
an optimizer mid-step. Ravex takes that moment from the `DataLoader` iterator,
so a loop over tensors that are already batched has to give it explicitly:

```python
for begin in range(0, len(data), batch_size):
    ravex.batch_boundary()
    ...
    optimizer.step()
```

Without it no round ever closes and every node trains alone for the whole run.
Ravex warns once, naming the call, a couple of steps after the first round
comes due.

**What it needs from the launcher.** Peer addresses and the job token come from
torch's rendezvous store, so the ranks must have called `init_process_group` —
`torchrun`, including across machines with `--rdzv-backend=c10d`. The process
group is used for its *store* and never for the exchange, so a rank dying
breaks a group nothing here touches; what it constrains is the start.

**`RAVEX_EXCHANGE_ADDRESS` is required whenever the nodes are not on one
network.** Behind NAT, nothing a process can ask its own kernel returns the
address a peer dials. Set it to the address peers should reach this node at,
optionally with a port.

**What `outer_inner_steps` costs the loss, measured.** A byte-level transformer
on two contiguous shards, 2048 local steps per node, held-out loss: H=1 **2.69**,
H=8 1.51, H=64 **1.45**, H=512 1.78 — against 1.55 for one node given the same
wall clock and 1.38 for one node given the same *tokens*. Two things to take
from it. **Small H is not the safe direction**: H=1 is the worst arm, because an
outer step per inner step compounds the Nesterov buffer at `outer_lr` thousands
of times and overwrites the inner optimizer each time. There is no small-H limit
where this becomes ordinary data parallelism. And **H=512 degrading is about
rounds, not about H**: at equal tokens a larger H is fewer exchanges, and 512
there was four of them in the whole run. Pick H from the link —
`bench/round_link_cost.py` reports the H at which the network is a quarter of
the round — and make sure the run gets more than a handful of rounds at it.

**`outer_save_dtype` is measured now, and still `null`.** Same bench: every
dtype column within ±0.005 of no cast at every H, with fp8 marginally ahead at
H=64 — noise, in other words. Including the
specific worry, that a delta truncated the same way every round accumulates its
error instead of averaging it out: at H=8 there are 256 rounds to accumulate
over and the column is flat. On the link this is for, bf16 took the report from
15.8 MB to 11.7 MB and the round's network from 3.28 s to 2.61 s — **a fifth
off every round** — for no measurable change in publish time, even though a cast
makes each node read its own report back before averaging it (local disk, not
network).

So it stays off not because nobody has looked, but because turning it on is what
surfaced a defect in the seed round, and because the evidence is one 0.48M model
on one box. Set `outer_save_dtype: bf16` and take the fifth — it is the smaller
of the two bets that measured free, keeping fp32's exponent range and losing
only mantissa (0.14% of relative error per element against fp8's 2.3%). `none`,
`off` and an empty value all turn it off again.

**What is not handled.** Floating-point buffers — batch-norm running statistics
— are not exchanged; each node keeps its own, and it says so once at startup.

## Diagnostics

One environment variable exists to make something measurable that is
otherwise unreachable. **Not a setting.** It has no place in `ravex.yaml`, it
does nothing useful in a real run, and it carries a rule about how it must be
applied across machines.

### `RAVEX_ASSUME_NO_NUMPY`

Forces Ravex to behave as though torch cannot convert a tensor to NumPy.

Every agreement between ranks — the step to resume from, which stores each
machine can see, the run id, whether the storage is shared, whether every rank
succeeded — goes through one object-gather helper. That helper has two
implementations, and it picks the NumPy-free one *only* when
`tensor.numpy()` fails. Every image worth renting ships NumPy, so without this
variable the replacement is unreachable on real hardware and would ship having
never run on a network.

**Set it per machine, deliberately.** The interesting case is one rank with
NumPy and one without, which is what two boxes from one provider look like
when they come up from different images. Setting it on one node and not the
other is the *intended* use.

`RAVEX_SPLIT_DRAIN` used to live here as a second diagnostic. It has been
promoted to the normal path — see [Reading the handoff
log](#reading-the-handoff-log) — and is no longer read.

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
