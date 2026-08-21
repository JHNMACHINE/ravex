# Changelog

## Unreleased

Jobs spanning more than one machine. Everything below exists because of one
question asked on 2026-08-18 — two boxes of eight GPUs, does this work? — and
the answer turned out to be "partly, and it does not tell you which part".

### Changed

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
