# Changelog

## Unreleased

### Added

- **Coordinated emergency checkpoint on SIGTERM, for sharded models split
  across more than one machine (GPU-92).**

  A preempted spot instance gets SIGTERM ~10s before it is killed. For a
  plain-replicated (DDP) job that was already enough: rank 0 holds the whole
  state and writes it without needing anything from anyone. For
  `sharded_checkpoints: per_rank`, it was not — extracting even one rank's own
  shard goes through PyTorch's own `get_state_dict`, which is collective even
  when nothing is gathered across ranks, so a lone preempted rank could not
  produce a checkpoint by itself. Until now the SIGTERM handler simply skipped
  the final checkpoint for every sharded job, unconditionally.

  This adds a minimal coordination channel: the rank that catches SIGTERM
  raises a flag; every rank checks for it at the same cadence
  (`emergency_check_every`, default every step); if any rank has it set, every
  rank enters the same collective save together, on the process group they
  were already going to use. Detection runs on its own short-timeout group,
  isolated from the one carrying gradient synchronization, so a stuck
  detection round fails in seconds rather than however long the training
  job's own process group is configured to wait. See
  `ravex._dist.collectives.emergency_group` / `emergency_signalled`, and
  `RavexRuntime._check_emergency_signal`.

  **Read this before relying on it.** The local handoff alone — no network,
  just copying state off the GPU and handing it to the backend — has been
  measured at 10.6s on 8× RTX 5060 Ti, against a SIGTERM budget of roughly
  10s. That number alone consumes essentially the whole budget before this
  channel's own cost. **This is a best-effort attempt that will often not
  complete, not a guarantee that spot training under FSDP is now reliable.**
  It exists because a save that sometimes lands is worth more than one that
  never does, not because the numbers were made to work.

  The channel is deliberately narrow — it says "save now" and nothing else;
  no rank ever waits for a new node or tears down its process group because
  of it. It is the minimal building block GPU-94 (elastic torchrun) will
  need to generalize for broader multi-rank coordination; it is not a
  substitute for that work; do not duplicate this pattern there.

  New checkpoints written this way carry `emergency: true` in their metadata,
  for recovery analysis — nothing reads it back to make a resume decision.

  Config: `emergency_coordination`, `emergency_check_every`,
  `emergency_timeout`. See
  [Emergency checkpoint on preemption](docs/configuration.md#emergency-checkpoint-on-preemption-sharded-models).

### Changed

- **A reshard no longer reads every old checkpoint twice (GPU-101).**

  Resuming onto a different number of ranks is two passes over the old stores:
  one to measure how long each old shard is, one to cut out the slices the
  plan asked for. Two passes is not the waste — it is what holds the memory
  bound, since knowing where a shard starts needs every length before it, and
  a single pass would mean every old snapshot resident at once.

  The waste was that the *measuring* pass loaded them. It only ever wanted
  lengths, and there was no way to ask a store for a shape: `load_step` was
  the only reader, and it materializes every tensor. So a reshard from N ranks
  read N complete checkpoints it needed and N it discarded — tens of gigabytes
  on a real model, and between machines it would have been that over the
  network, which would also have undone the `ceil(N/M)+1` bound the reshard is
  built to hold.

  `CheckpointBackend.describe_step` is the question that was missing: the same
  tree, with each tensor replaced by its shape and dtype and everything that
  was never a tensor — placements included — left as it is. On Moonclip that
  is two small reads of one template entry and no tensors at all.

  Nothing depends on it being available. A backend that cannot describe itself
  returns `None` and gets loaded, which is exactly what happened before; the
  `torch_save` backend does precisely that, having one pickle and no way to
  read a shape out of it short of unpickling the lot. Needs Moonclip >= 0.1.0
  for the fast path.

## 0.0.5 — 2026-08-30

### Added

- **`save_dtype`: what precision each part of a checkpoint is stored at.**

  ```yaml
  save_dtype:
    optimizer: bf16
  ```

  Keys are component names — `model`, `optimizer`, `scheduler`, `scaler`,
  `dataloader` — or raw Moonclip globs for anything they do not cover. **The
  first matching rule wins**, in the order written, so an exception is
  expressed by putting it first: `{model: none, "*": bf16}`. A bare value
  applies to everything. `RAVEX_SAVE_DTYPE=model:none,optimizer:bf16` from the
  environment, ordered left to right.

  It is the largest saving available on a run that checkpoints often. On a
  1.5B model under FSDP2, per rank: the weights are a third of the state and
  delta well, −70%; `exp_avg` and `exp_avg_sq` are the other two thirds and
  delta essentially not at all, −1.1% and −4.0%. With β₁ = 0.9 a tenth of each
  moment is replaced by fresh gradient every step, which moves nearly every
  mantissa bit, so the XOR between two steps is noise. **85% of the bytes
  written are the part that does not compress** — and it is also the part that
  tolerates the least precision, since `exp_avg_sq` reaches Adam through
  `sqrt(v)`, which halves the relative error. `{optimizer: bf16}` halves 85%
  of the volume and leaves the model exactly as it was.

  **Components rather than patterns, and the reason is not tidiness.** An
  optimizer is written in two places: under `ravex/optimizers/…` when it is
  not sharded, and under `ravex/sharded/<key>/optimizer/…` when it is, because
  sharded state has its own collection path. A pattern written by hand as
  `ravex/optimizers/*` is right on one GPU and casts **nothing** on the
  sharded run the setting was chosen for — no error, no warning, a
  full-precision checkpoint. Naming the component expands to both. Raw
  patterns still work and pass through untouched, and there is a test that
  documents the trap rather than pretending it is unreachable.

  **Off by default, and it stays off across an upgrade.** It changes the
  numbers a resumed run gets back, and a library that did that to an in-flight
  run because someone bumped a version would be wrong to. What it does instead
  is log one line per run naming the setting when it is unset — at `INFO`, not
  as a warning, because nothing is wrong; the setting is simply worth a great
  deal and is invisible otherwise.

  Values are checked when the config loads rather than when the checkpoint
  manager is built. Moonclip refuses a bad dtype too, but it does so at
  construction, which under Ravex is mid-run and inside the `except` that
  falls back to `torch.save` — so a typo would have cost the run its Moonclip
  checkpointing and said one line about it. A bad rule is now dropped, that
  component is stored unchanged, and the reason lands in the startup problem
  report with everything else. One bad rule does not take the correctly
  spelled ones with it.

  Needs Moonclip 0.0.9, which is already the floor.

  And it says so rather than crashing into it. Moonclip before 0.0.9
  declares `save_dtype` as a string, so the mapping form reaches it as a
  `TypeError` — which `get_backend` catches along with everything else,
  so one configuration line would have cost the run its Moonclip
  checkpointing entirely, reported as "Moonclip backend unavailable". That
  is the exact failure the 0.0.8 post-mortem named: a version requirement
  wearing compatibility as a disguise. The version is now checked in the
  open, and what degrades is the option rather than the backend —
  checkpointing continues, at the precision the tensors arrived in, and
  the log line says which Moonclip would be needed.

- **`reshard_on_resume`: a `per_rank` checkpoint can be resumed at a different
  world size.** Each rank rebuilds its own shard out of the old ones — eight
  ranks' shards onto four, or four onto three, where no new shard equals any
  old one and each is stitched from two.

  The offsets are *measured*, never recomputed. The old lengths are the shapes
  of the tensors that were saved; the new ones are the shapes the live model is
  holding, shared in one `all_gather_object` of a few integers. Reproducing
  torch's chunking rule to derive them would have been a second implementation
  of a rule with an uneven-tail case, and it would have drifted from the first
  in silence.

  **Off by default**, and not out of caution about the arithmetic: a resume
  that reshards silently is a resume that silently succeeds when the launcher
  started three ranks where the job wants four. The mismatch is detected and
  logged always; acting on it is the opt-in.

  Refused, loudly and by name, when a precondition does not hold: a 2-D mesh
  (FSDP crossed with tensor parallel) is a cartesian problem rather than an
  interval one and is not attempted, and a reshard whose old shards are not all
  reachable — as stores or as *complete* peer copies — stops rather than
  assemble tensors with a band of uninitialised rows in them. Local storage
  only so far; on remote the mismatch is reported and the run starts clean.

  Two things do not reshard, and are documented rather than left to be found.
  The sampler partitions the epoch by world size, so a resumed run continues
  the model and not the run: the position is rescaled to preserve the total
  data consumed, every sample is still seen once per epoch, and the order is
  not the one the original run would have taken. And there were N per-rank RNG
  states where there are now M ranks, with no correct mapping between them, so
  they are not restored and the log says so.

  Peak memory is one old snapshot plus this rank's new shards. The old stores
  are read twice — once to measure, once to cut out the slices the plan asked
  for — because a single pass would mean holding every old snapshot at once,
  which is the whole checkpoint per rank. The global tensor is never
  materialised anywhere.

### Changed

- **A diagnostic, `RAVEX_ASSUME_NO_NUMPY`.** Not a setting: it has no place
  in `ravex.yaml` and does nothing useful in a real run — it exists to make
  something measurable that is otherwise unreachable. See
  [Diagnostics](docs/configuration.md#diagnostics).

  It forces the NumPy-free object-gather path, which no ordinary image
  reaches because they all ship NumPy — without it that code would have
  shipped having never run on a network. It is *meant* to be set on one
  machine and not the other: a rank with NumPy and a rank without is what
  two rented boxes from different images look like.

- **The checkpoint handoff always reports `skew` apart from `drain`, on a
  sharded run with more than one rank.** A barrier now runs before the
  accelerator drain unconditionally in that case, timed apart, separating a
  rank waiting for its own device from a rank waiting for a peer.

  This was `RAVEX_SPLIT_DRAIN`, an opt-in diagnostic added earlier in this
  same release. It has been promoted to the normal path rather than kept
  behind a flag, because what it measured on two machines settled the
  question it was written to ask: `skew 34.6 s / drain 0.000 s` on one rank
  and `skew 103.8 s / drain 0.000 s` on the other — all of what used to be
  called `drain`, and excluded from the reported cost on the theory that it
  was queued compute, was in fact a rank waiting on a peer that was late
  because *it* was checkpointing. See GPU-98.

  `drain` keeps its exclusion — genuine queued device work still is not a
  cost of checkpointing. `skew` does not: it is counted like every other
  phase, which means the cadence warning (`checkpoint_every=... is costing
  N% of wall time`) can now fire on a run where it previously stayed quiet,
  because time that was always being spent is no longer hidden under a name
  that excluded it. That is the point of the change, not a side effect of
  it.

  The guard is `sharded and get_world_size() > 1`, computed identically on
  every rank — the same condition the save's own verdict collective already
  carries. A replicated (non-sharded) job never reaches the barrier, for the
  same reason it never reached the diagnostic's version of it: the non-main
  ranks are gone by then, and a barrier only rank 0 entered is a hang until
  NCCL gives up. `RAVEX_SPLIT_DRAIN` is no longer read; setting it does
  nothing.
- **Python 3.9 and 3.10 are no longer supported.** The floor is 3.11, and the
  CI matrix runs 3.11 through 3.14.

  The list is [the Python devguide's](https://devguide.python.org/versions/)
  rather than a judgement of our own about who is still out there. 3.9 reached
  end-of-life on 31 October 2025, ten months before this release. 3.10 is
  still in security-only maintenance, but it reaches end-of-life in **October
  2026** — two months from now — so stopping there would have meant doing this
  again almost immediately. 3.11 runs to October 2027.

  Dropping 3.9 also removes a real cost rather than a nominal one. On 3.9 a
  clean `pip install torch` cannot `import torch.distributed.checkpoint` at
  all: `torch.distributed.elastic.rendezvous.registry` does `from
  importlib_metadata import entry_points` under a `sys.version_info < (3, 10)`
  guard, and the wheel's metadata declares only filelock, fsspec, jinja2,
  networkx, sympy and typing-extensions. Verified against the CPU wheel on
  `python:3.9-bookworm`: the import fails, and installing that one backport
  fixes it. The sharded-resume tests died there and on no other interpreter.
  Every line of it is gated on being below 3.10, so raising the floor deletes
  the problem instead of carrying a workaround for it.

  3.15 is **not** in the matrix, and was for exactly one day. It went in as a
  release candidate marked `continue-on-error`, on the theory that a prerelease
  is worth watching early and never worth holding a tag for. The theory did not
  survive contact: torch on 3.15.0rc1 segfaults in its own C++ —
  `TensorImpl::incref_pyobject`, reached through `torch.save` — and takes the
  whole pytest process down on the *first* test. So there was no partial signal
  to watch: 246 collected, nothing run. What was left was a permanently red
  square, which is the thing a real red hides behind on the day someone checks
  the matrix before tagging. It comes back when torch survives 3.15.

- **Shard placements are stored as data, not as `str(placement)`.** What went
  into a checkpoint was torch's own short repr — `S(0)`, `R`, `P(sum)` — and
  the only way back from it is a parser for a string nobody promised to keep.
  Resharding has to ask which dimension a tensor was split along, so the answer
  is now written down as an answer. The mesh shape is recorded alongside.

  Checkpoints written before this are still read: both the short form and the
  long `Shard(dim=0)` one are parsed back. Nothing about resuming an existing
  checkpoint at the same world size changes.


- **The Moonclip backend now builds `MoonclipManager` rather than
  `CheckpointManager`.** Ravex owns the topology and hands Moonclip a value;
  it no longer relies on a convenience layer that worked one out.

  `CheckpointManager` read `RANK`/`WORLD_SIZE` from the environment whenever
  it was not told. Under `torchrun` it would then believe it was one rank of
  eight and reject the single-rank save API outright — and this backend caught
  that in the `except` around its own construction and fell back to
  `torch.save`, with one log line to say so. A distributed run lost Moonclip
  checkpointing and kept training. The pinning that avoided it has been in
  place since the failure was found, and a test has guarded it since; what
  changes now is that there is nothing left to pin against.

  `MoonclipManager` is the explicit layer underneath, it never guessed, and it
  already accepts every option this backend passes. Nothing changes about
  where checkpoints land or what is in them.

  Requires Moonclip **0.0.9**, and the floor moves accordingly — not for the
  manager, which has always been there, but for `unflatten_state_dict` below.
  A version requirement stated in `pyproject.toml`, rather than a method call
  wrapped in a guard that answers "nothing here" when the method is missing.

- Loading goes through Moonclip's `unflatten_state_dict`, new in 0.0.9 and the
  public inverse of the `flatten_state_dict` this backend already used to
  write. `CheckpointManager.load` rebuilt the state tree *and* called
  `load_state_dict` on live objects; this backend has no live objects at that
  point and wanted only the first half. Until 0.0.9 the first half had no
  public name, so taking it meant taking the second as well.

- **The checkpoint handoff reports `backpressure` apart from `store`.** They
  were one number, and the sum reads like the writer.

  `store` is the shadow copy — the reason the training loop can carry on
  mutating weights while the writer is still working. It is memory bandwidth,
  it scales with the model, and it comes down by making the state smaller.
  `backpressure` is the wait for the *previous* checkpoint's writer: Moonclip
  allows one save in flight, so a writer that has not drained stops the next
  save before any of its work starts. It scales with the cadence and with the
  storage, and it comes down by checkpointing less often or writing somewhere
  faster.

  Added together they name neither. Watching the combined figure grow is what
  produced a hypothesis about a writer in deficit, several A/B runs, and an
  issue filed against the writer — while the phase was a 2 GiB `memcpy` for
  the whole of it. On a 1.5B model at cadence 2 the wait was 4-12% of what the
  single number showed.

  Now, on a run whose writer is genuinely behind, the log line says which:

  ```
  step 2: flatten 0.001s, store 0.101s, backpressure 1.655s
  ```

  where before it said `store 1.756s` and left the reader to guess. Needs
  Moonclip 0.0.9, whose `last_queue_wait()` is where the number comes from —
  it existed before, but only as a line `MOONCLIP_PROFILE=1` printed to
  stderr, which is not somewhere a program can read it. The cadence alarm
  counts the new phase like every other one: it is wall time the training loop
  pays, and it was already in the total.

### Fixed

- **Torch without NumPy broke every collective on the resume path.** Ravex
  depends on PyYAML alone, and torch does not require NumPy either, so an
  install with neither is a supported way to run this. But
  `dist.all_gather_object` decodes what it gathered with
  `tensor.numpy().tobytes()`, so on that install every agreement between ranks
  raised `RuntimeError: Numpy is not available` — not at import and not at
  startup, but at the first collective of a resume, after the wire work had
  already happened and with nothing left for the caller to fall back to.

  The five object collectives — the step every rank holds, the stores each
  machine can see, the run id, the shared-storage probe and the all-or-nothing
  resume flag — now go through one wrapper. Where the conversion works it is
  still torch's own call. Where it does not, the wrapper posts the same two
  `all_gather` calls in the same order and with the same shapes and dtypes,
  and decodes with `bytes(tensor.tolist())`. Matching torch on the wire is the
  point of that care: a job whose ranks disagree about NumPy still meets,
  which is checked with two gloo ranks, one of each.

  The probe is `tensor.numpy()` itself, asked once, rather than `import
  numpy` — torch initialises NumPy on its own, and a version it was not built
  against imports perfectly well and then fails the conversion.

  Found by CI rather than by a user: the unit job installs torch and pytest
  and nothing else, which is exactly this configuration.

- **A store recovered over the network stayed marked as a copy.** A rank that
  came up without its own store and took one back from a peer was left holding
  `.ravex-replica-ok` in its own store directory. Promoting a copy off this
  machine's own disk removed it; the two roads home ended differently.

  The cause is that `StoreWriter` writes the marker into whatever directory it
  filled, which is right — from where it stands it is always building a copy,
  and while the bytes are arriving the absence of that file is the only thing
  separating a half-built store from a whole one. What was missing is the
  other end of the same act: `promote_copy` removes it deliberately once the
  store is whole, and the network path did not.

  No known consequence today — `replica_is_complete()` is asked about replica
  paths, not about a rank's own store, so nothing read the file where it
  should not have been. It is worth correcting because the claim goes false
  immediately: it says a copy is complete about a directory that is, from the
  next save onward, a live store being rewritten. True for one instant and
  left where a later reader would believe it.

  Found on the first pair of real machines this project has run on (two RunPod
  pods, 2026-08-21). Nothing compared the two paths, which is why a divergence
  this small reached rented hardware; a test now holds them side by side.

- **A shard rebuilt on an older torch could come back with a global shape
  nobody asked for.** `DTensor.from_local` is given `shape=`/`stride=` so
  uneven shards survive; a torch whose signature predates those arguments
  raises `TypeError`, and Ravex fell back to calling it without them. That
  fallback infers the global shape as local × mesh size — correct when the
  tensor divides evenly across the mesh, and wrong by exactly the remainder
  when it does not. A 10-row tensor over 4 ranks came back claiming 12.

  Nothing failed at the point of the mistake. Every shard was individually
  valid, `set_state_dict` accepted them, and the first symptom would have been
  somewhere else entirely — which is the same shape as GPU-59 and GPU-79, and
  the reason this is worth a release note rather than a line in a diff.

  The fallback is now taken only after checking that the shape it would infer
  is the shape that is actually there, and refuses with both numbers when it
  is not. Even splits are unaffected, which is most of them. Every rank
  reaches the same verdict — with an uneven split the ranks holding a full
  chunk infer too much and the short one too little, so none of them agrees —
  so no rank is left inside a collective while another raises.

### Removed

- **Four compatibility guards against Moonclip builds the floor already
  excludes.** GPU-88 asks for this sweep before every tag, and names the shape
  to look for: *every guard around a backend call is a version requirement in
  disguise*. Each one below was dated against the release that introduced what
  it guarded, and every one of them was unreachable.

  - `restore_from_remote` was wrapped in `except AttributeError` returning
    `False`. This is the exact case GPU-88 was written about. The method has
    existed since Moonclip 0.0.8 and the floor has said `>=0.0.8` since — the
    guard simply outlived it. It never crashed; it answered *"nothing to
    restore"*, which is indistinguishable from an empty bucket. A rank whose
    disk had been replaced would have started from scratch with its own data
    sitting in the remote, and taken every other rank back with it, because a
    resume is agreed at the oldest step everyone holds.
  - `flatten_state_dict(as_tensors=)` was feature-detected, with the byte path
    as a fallback and a log line saying checkpoints would block the training
    loop about five times longer. The parameter shipped in 0.0.4.
  - `keep_base_in_memory` was feature-detected, warning that the base would be
    retained anyway when a run had explicitly asked otherwise. It shipped in
    0.0.4 too.
  - `_accepts`, the introspection helper the last two used, with nothing left
    to ask.

  None of the removed branches had a test. That is not a coincidence: a branch
  that only runs against a build the packaging forbids cannot be exercised
  without installing one.

  One guard of that shape survives, because it is not dead: the
  `except TypeError` around `DTensor.from_local(shape=, stride=)`. Torch is
  not a declared dependency — Ravex attaches to whatever build is already
  installed — so there is no floor to date it against. What it *did* do
  silently is fixed below.

## 0.0.4 — 2026-08-21

Jobs spanning more than one machine. Everything below exists because of one
question asked on 2026-08-18 — two boxes of eight GPUs, does this work? — and
the answer turned out to be "partly, and it does not tell you which part".

On 2026-08-21 the question was finally put to two machines that were actually
two machines, on a network that could actually fail. The replication holds:
a box that arrived with an empty disk took its store back from its peer and
resumed where the run had stopped, and a copy caught half way was refused
rather than trusted. What that day changed in this release is the last two
entries below — both are things a single machine could not have shown, because
on loopback the number they hide is zero.

### Changed

- **The handoff breakdown names the replication, and the cadence observation
  counts it.** `Checkpoint at step N handed off in 186.362s (drain 34.898s,
  collect 0.228s, flatten 0.001s, store 0.030s)` — a line that declares 186
  seconds and explains 35. The missing 151 were the copy going to the other
  machine: timed inside the handoff, named by nothing, and therefore invisible
  to the observation that exists to say *"checkpointing is eating your wall
  time"*. That observation sums the phases this line names, so it computed
  0.26s against a 291s interval, called it 0.09%, and stayed silent while the
  run spent 64% of its wall time stopped.

  Over loopback the copy costs milliseconds, which is why a missing phase hid
  a number that was always zero. It took two machines with 100 Mbps between
  them — measured at 7 MB/s each way — for it to become the whole handoff.
  Now there is a `replicate` entry, the phases add up to the total, and the
  same situation reports **42% of wall time** and says so. `drain` is still
  excluded, and for the reason it always was: it is the training loop's own
  queued work coming due, not a cost of checkpointing. Replication is not that
  — without it the time would not exist at all.

- **A copy caught mid-transfer is now named rather than called absent.** Losing
  a machine while a replica is in flight left the survivor saying *"No store
  anywhere for rank(s) 1 - starting from scratch. Either the run that wrote
  these had fewer ranks, or the machines holding the last ones are gone"*.
  Neither was true: the machine printing it was powered on with 680 MB of that
  rank's store on its own disk, disqualified because `StoreWriter` removes the
  completeness marker before the first byte lands and the transfer never
  finished. The refusal is correct — a torn copy is not a checkpoint — but the
  explanation sent the reader looking for hardware that was fine.

  The machine holding the copy now says so. It is the only one that can tell
  the two apart, and the others go on reporting what they see. Not a rare
  corner on a slow link: measured at 0.5s resolution on the same pair, a
  ~670 MB store takes 102s to copy against a 244s cycle, so the copy is
  unusable **42% of the time** and losing a machine during a transfer is close
  to a coin toss.

- **Peer replication now moves a store at the speed of the wire.** Copying a
  checkpoint to another machine was running at about a third of what the same
  link carried with nothing else in the way: 529 MB/s against 1462 MB/s on a
  rented box, and the same shape on a laptop. It was not the network and not
  the disk — reading and framing the store ran at 3411 MB/s on its own. It was
  four copies of every chunk, two on each side, all of them holding the GIL.

  Two of them are gone. The sending side hands `isend` a tensor over the
  chunk's own memory instead of a `bytearray` duplicate — safe because
  `fixed_chunks` yields fresh immutable `bytes`, which is the only thing the
  copy was protecting against. The receiving side writes straight from the
  tensor's memory when nothing is half-parsed, instead of `tobytes()` into a
  `_pending` buffer, slicing a piece out, and memmoving the tail. Headers and
  file boundaries still take the buffered path, which is where the ragged
  cases always lived.

  Measured on the same machine, alternating arms: **355 MB/s → 527 MB/s, a 48%
  improvement**, which puts the transfer at the wire's own rate. A threaded
  version that also overlapped disk with network was tried and added only six
  points on top — not worth concurrency in a recovery path, so it was dropped.
  The bytes on the wire did not change: a patched peer and an unpatched one
  still understand each other.

- **`pip install ravex` now installs the autoloader.** The one-line
  `ravex_autoload.pth` ships in the wheel, so checkpointing works on a project
  with a `ravex.yaml` without anyone running `ravex enable` first. The README
  used to promise the opposite — *"installing the package changes nothing on
  its own"* — and that sentence is gone.

  What replaces it is a weaker promise that is worth more, because it is about
  what the file *does* rather than about its absence: **Ravex is inert until a
  project asks for it.** The line runs in every interpreter in the
  environment, imports only `os` and `sys`, looks for a `ravex.yaml` at or
  above the working directory, and — finding none, and no `RAVEX_ENABLED` —
  installs nothing at all and returns. It never imports torch to decide. When
  it does arm, it arms a hook that waits for `import torch` and loads the
  runtime only then, removing itself once it has fired.

  Measured with `-X importtime`: **1.8 ms**, down from 11.6 ms once `typing`
  was taken off the path (see below). Verified on a real wheel rather than on
  the configuration — the file lands in `site-packages`, not in the
  `.data/` directory that `data_files` would have put it in and that pip
  installs one level above where a `.pth` is ever executed. It is recorded in
  `RECORD`, so `pip uninstall ravex` takes it away again: an orphaned `.pth`
  importing a module that no longer exists is the classic way this scheme
  breaks, and it is not a risk here.

  `ravex enable` still exists, for putting the file back after
  `ravex disable`. `ravex disable` now says that a later
  `pip install --upgrade` will restore it.

- **`ravex/__init__.py` no longer imports `typing`.** It was 9.8 ms of the
  11.6 ms the autoloader cost — 85% — for three names used only in
  annotations, which the `from __future__ import annotations` already at the
  top of the file turns into strings that are never resolved. The annotations
  now use builtins. Visible in the public typed surface: `status()` returns
  `dict[str, object | None]` rather than `Dict[str, Optional[Any]]`, and
  `object` is stricter than `Any` for a consumer.

- **The Moonclip floor is `>=0.0.8`.** Ravex calls `restore_from_remote()`,
  which the engine grew in 0.0.8, and the backend catches the `AttributeError`
  an older one raises. That is the right thing for a method that may not be
  there, and it is also why the pin has to move: on 0.0.7 the call does not
  fail, it answers "nothing to restore" — so a rank whose disk was replaced
  starts from scratch with its own data sitting in the bucket, and takes every
  other rank with it. The code path cannot tell an empty remote from an engine
  that has no way to read one; the version requirement can. The same release
  is also the first that can put an object over 5 GiB into S3 at all, which
  `sharded_checkpoints: gather` reaches on its own.

### Added

- **The storage topology is announced at activation.** With
  `sharded_checkpoints: per_rank` and local storage, each machine writes only
  its own ranks' shards to its own disk, so no machine holds a whole
  checkpoint. It then resumes only if every machine is handed the same ranks
  again — which no launcher promises — and not at all if a machine is lost.
  The behaviour was safe and silent: the run simply started over, with no
  indication that a checkpoint existed and had not been used. Ravex now says
  which of three situations you are in, at activation rather than at the first
  failed resume.
- **Shared storage is probed, not guessed.** A local disk and an NFS mount are
  the same `type: local` pointing at a directory that exists, so the question
  is asked instead: every rank drops a uniquely named marker and looks for
  everyone else's, with `all_gather_object` as the synchronisation. A directory
  that cannot be written to answers "not shared", which is the safe reading.
- **`replicate_every`** — copies of each rank's store to a peer on another
  machine, every N checkpoints, for jobs with neither a bucket nor a shared
  filesystem. The peer is `(rank + local_world_size) % world_size`, which only
  lands on a different machine when the ranks are spread evenly, so an uneven
  layout is reported as *not* replicating rather than assumed to work. The
  exchange is point-to-point `isend`/`irecv`, never a collective — an
  all-gather would leave every rank holding `world_size` copies. All-or-nothing
  with a collective verdict: three copies of four landing is not a restore
  point and is not recorded as one. Replicas live under `replica/`, outside the
  `rank_*` names discovery scans, and count only once a completion marker is
  written last. Default 10; `0` turns it off. **The guarantee, stated exactly:
  the loss of any one machine is survivable, at a cost of at most one
  replication interval of progress.**
- **A replaced machine gets its store back.** A rank that comes up with nothing
  pulls its store from the peer that has been holding a copy, or — with a
  bucket configured — from the remote. Without this the copies existed and
  nobody read them, which protects the bytes and not the run.
- **Run identity and owner records.** Each per-rank store carries a
  `.ravex-owner` naming the run that wrote it and the machine it was written
  on. Two training histories on one disk used to be indistinguishable: on
  2026-08-19 a run restarted with a different placement wrote a second history
  beside the first, and a later restart resumed the accidental one while the
  original sat one directory away. It is also what turns "rank 2's store is not
  here" into "rank 2's store was written on node0, which is not running rank 2
  now".

- **A run is told when its cadence is expensive.** Once, and as a statement of
  what happened rather than a prediction: what fraction of wall time went into
  checkpoint handoff between the last two checkpoints. Measured on 8x RTX 5060
  Ti with a 1.5B model, `checkpoint_every=2` spends a third of wall time on
  handoff and nothing breaks — the writer keeps up, the run is simply slower
  than its author probably meant. So it does not claim the cadence is
  unsustainable and it does not quietly raise it; both would be guesses about
  a machine the process cannot see, while the ratio is a fact it can. The
  `drain` phase is left out, being the training loop's own queued GPU work
  coming due rather than a cost of checkpointing.

### Fixed

- **Replication never worked on a GPU job, and said it did.** Moving a store
  between machines means moving bytes, and bytes live on the host — but the
  transfer went out on the **default** process group, which on a multi-GPU job
  is NCCL. NCCL is a GPU collective library and refuses host tensors outright.

  Measured on 8x RTX 5060 Ti on 2026-08-21: every replication round failed
  with `No backend type associated with device type cpu`, **not one copy was
  ever made**, and the announcement at activation went on promising that
  losing a machine would cost at most one interval. The failure was reported —
  a warning per round — but the promise was louder and came first.

  Bytes now travel on a **gloo** subgroup, opened once at activation and
  reused. On a job that is already gloo the default group carries host tensors
  perfectly well and no second group is made. Where neither is possible,
  replication is declared **off at activation**, in the same message that
  would otherwise have promised it — because a run that is not protected
  should be told once, at the start, rather than a warning at a time into a
  log nobody is reading.

  Verified on the same hardware after the change: six rounds out of six, and
  both disks holding the peer's copy where before they held nothing.

  Nothing on CPU was affected, which is exactly why nothing caught it: the
  four-scenario container bench runs on gloo, where the same code is correct.
  The backend was the one variable it could not vary.

- **A bucket was a backup you could not resume from, and the documentation
  said otherwise.** Moonclip's remote support was push-only, so the step to
  resume from was read from the *local* manifest. On a six-node bench a node
  whose disk had been replaced started from scratch with its own data sitting
  in the bucket, and took every other rank with it, since a resume is agreed at
  the oldest step everyone holds. Fixed in Moonclip; Ravex now fills an empty
  store from the remote before deciding where to resume. The runtime warning
  that used to recommend S3 for a problem S3 did not solve was corrected in the
  same pass.
- **One rank disabling itself no longer hangs the other seven.** A failed
  checkpoint called `_disable`, which is per process: a full disk on one node
  of eight turned that rank off while the rest stayed on. At the next
  checkpoint the seven entered `collect_state` — a collective — and the eighth
  returned at the first line and ran ahead into the next forward. The seven
  then waited for a participant that never came, and NCCL takes half an hour to
  say so, with all eight GPUs allocated and billing throughout. Not a fast
  error, an expensive hang. The verdict is now taken once, by everyone, and
  everyone acts on it. A backend that is merely unavailable is reported rather
  than latching the runtime off.

- **After a reshuffle every copy was present and none was used.** The ring
  addresses a peer **by rank**, but a copy travels **with the disk** it was
  written to. Move the nodes round by one position and each rank comes up
  sitting on the copy of its own store — complete, and unreachable, because
  the peer that used to hold it is now elsewhere holding somebody else's. Six
  intact copies bought nothing, and the run started from scratch with the
  bytes under its feet. A rank without a store now looks for a copy of itself
  on its own disk first: a local file copy, no pairing and nothing on the
  wire. It is promoted only when the history it belongs to is unambiguous —
  named by the stores that survived, or, when none did, agreed among the
  copies themselves. A copy from an older run is refused, because resuming
  half the shards from one training history and half from another is a wrong
  model and a silent one.

### Testing

- **Two scripts for the replication transfer**, in `integration/scripts/`, both
  running two gloo processes with no GPU and no rented machine — the path was
  measured at 529 MB/s over loopback and 562 MB/s over real TCP between
  containers, so whatever binds it can be studied on a laptop.

  `measure_replication_transfer.py` times the transfer against the two bounds
  it has to be read against: the wire carrying the same volume with no file
  touched, and the disk delivering the store with nothing sent. When the
  transfer sits far below both, its `phases` arm says which step is spending
  the time. That is how the 48% came off.

  `verify_replication_transfer.py` sends a deliberately awkward store — a
  zero-length file, a one-byte file, a file exactly one chunk long, another one
  byte past the boundary, nested directories — and compares the arrival with
  the source by hash. The unit tests drive `StoreWriter` in one process; this
  is the only thing that exercises `exchange_stores` between two of them.

- **A multi-machine bench**, `integration/multinode/`. One container per rank,
  because the question is which ranks can see which directory and ranks on one
  machine all see the same one — `torchrun --nproc_per_node=6` makes six
  processes, not six machines. Four scenarios: shared storage, split storage,
  a machine replaced with an empty disk, and the nodes coming back in a
  different order. It prints what each disk holds before the resume, so a run
  can tell "the data was gone" from "the data was there and nobody looked".

  It is the first end-to-end exercise of both the shared-storage path and the
  peer replication, and it earned its keep on its first honest run by finding
  the reshuffle defect fixed above — which no unit test could have found,
  because every piece involved was correct on its own.

  Worth knowing if you extend it: `--init` is load-bearing. A round ends with
  the training script sending itself SIGKILL, and the kernel discards a
  SIGKILL aimed at PID 1 from inside its own namespace when PID 1 has no
  handler. Without an init process the script runs to completion and the bench
  reports six happy nodes having proved nothing.

### Documentation

- **The multi-machine story is written down.** It had never been: `multi-node`,
  `nnodes` and `NFS` appeared nowhere in `docs/`, the README or the sources,
  and a runtime warning is not a substitute for a page you can read before
  starting a job. See *More than one machine* in
  [docs/how-it-works.md](docs/how-it-works.md) and
  [docs/configuration.md](docs/configuration.md).
- **Keeping the checkpoints after a run ends is the user's**, stated as such,
  together with where to look: a sharded model has no checkpoint at exit, so
  the newest thing worth copying off is the last periodic one.

## 0.0.3 — 2026-08-18

**No changes to the library.** `git diff v0.0.2..v0.0.3 -- ravex/` is empty: the
package you install is byte-for-byte the 0.0.2 one. This release exists to carry
the packaging fix below and to put the release pipeline through a real
publication for the first time, which is a thing worth knowing about a version
before you wonder what it changed.

### Fixed

- **The source archive contains the changelog again.** Setuptools takes the
  readme and the licences from `pyproject.toml` but has no field for a changelog,
  so `ravex-0.0.2.tar.gz` shipped without one — the sdist being exactly the copy
  a distribution packager or an auditor reads, and the one that survives if the
  repository does not. A line of `MANIFEST.in` fixes it; this is the first
  release built with it, and the archive was checked rather than assumed.

### Infrastructure

Not shipped, but this is the release where CI started meaning something. Every
job had been failing: `--index-url` for the CPU torch wheels *replaces* PyPI
instead of adding to it, so pip could not find the build backends torch's own
dependencies need and the six interpreter jobs died in six seconds each. The
integration job hit the ten-minute runner limit to the second, and now runs in
under five with the framework tests split into a job of their own. One unit test
asserted Windows path semantics on a POSIX interpreter and had never run
anywhere it could fail.

## 0.0.2 — 2026-08-17

First release on PyPI: `pip install ravex`, or `pip install "ravex[moonclip]"` on
Linux for the delta-tracking engine. Alpha, and the limits below are the part
worth reading before you rely on it.

### What it does

Transparent checkpoint and resume for PyTorch training: no change to your script,
not one import. A killed run, restarted with the same command, continues with the
same weights, optimizer moments, LR schedule, AMP loss scale, RNG state and
position in the dataset. The test suite asserts the strong form — a run killed at
step 20 and resumed produces losses bit-identical, step by step, to the run that
was never interrupted.

Verified: plain loops, gradient accumulation, LR schedulers, AMP, `num_workers >
0`, DDP over NCCL, FSDP1 and FSDP2, and a killed `torchrun` job resuming on every
rank. The CUDA paths were last exercised on 8× RTX 5060 Ti with torch 2.12/cu130.

### Added

- **Per-rank sharded checkpoints** (`sharded_checkpoints: per_rank`). Every rank
  writes its own shard into its own store instead of gathering the whole state on
  rank 0. On a 1.48B model with Adam, collecting the state went from 15.6 s to
  1.5 s, and peak host memory from 18.1 GiB on rank 0 to 6.1 GiB on every rank
  alike. The default stays `gather`, because per-rank shards only resume at the
  world size that wrote them and losing that is not something you should acquire
  by upgrading.
- **The per-checkpoint log line reports where its time went**, not just the total:
  `handed off in <total> (collect …, flatten …, store …)`, or `copy`/`queue` on the
  `torch_save` backend. A handoff is time the training loop stands still, and the
  total alone never says which phase to look at — the last investigation of a slow
  one cost four experiments that each ruled out the wrong suspect.

### Fixed

- **The `torchrun` launcher no longer starts a runtime.** It imports torch to
  parse its own arguments, so it came through the autoloader like everything else:
  a run with eight ranks announced nine runtimes, the extra one being a process
  that never trains. Harmless in practice — it holds no model, so it never
  reached the storage backend — but harmless by construction rather than by
  design.
- **The version was declared in two places** and diverged at the first bump: the
  CLI announced 0.1.0 while the package was 0.2.0. `ravex/__init__.py` is the only
  source now, and the release refuses to publish a tag that disagrees with it.

### Known limits

These are choices, not defects, and each one costs something specific.

- **No final checkpoint at exit for a sharded model.** Shutdown is exactly where
  ranks stop being in lockstep, and a collective nobody else joins does not raise
  — it hangs. Losing the last few steps is a bounded cost; a hang on rented
  hardware is not. Set `checkpoint_every` accordingly.
- **`IterableDataset` position is not replayed.** There is no index sampler, so
  the position in the stream cannot be reproduced. Everything else is restored.
- **With HuggingFace `Trainer` or Lightning, state restoration is exact but replay
  is not.** Both iterate the dataloader on their own schedule and consume the
  global RNG around the loop, so a resumed run continues correctly from the
  checkpointed state and then sees a different shuffle. Plain loops, DDP and FSDP
  are bit-exact with randomness on; this is a framework-interaction limit.
- **Your loop's bounds are still yours.** A resumed script runs its own
  `for epoch in range(N)` from the top and has no idea 3000 steps already
  happened. Set `max_steps` and Ravex ends the run at the right step however many
  times the process restarted.
- **`auto_wrap_policy` is not optional for FSDP1** at real sizes: without it the
  model is one flat unit and every rank materialises it whole in the forward pass.
  A 1.5B model OOMs on eight 16 GiB cards.
