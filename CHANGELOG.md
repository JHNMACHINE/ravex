# Changelog

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
