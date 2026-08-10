# Ravex

Transparent checkpoint and resume for PyTorch training. Your training script
does not change — not one line, not one import.

```bash
pip install ravex
ravex enable
```

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

## Two ways to use it

**Zero code changes.** `ravex enable` installs a one-line `.pth` file in
site-packages, which Python executes at interpreter startup. From then on Ravex
attaches itself to any training process that has a `ravex.yaml`.

**One line**, when you would rather be explicit:

```python
import ravex
ravex.activate()
```

Both do the same thing. The `.pth` route exists so that a platform can enable
checkpointing for code it does not own.

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
- installing the package changes nothing on its own. Without `ravex enable`
  there is no `.pth`; with it, Ravex still only wakes up for projects that have
  a `ravex.yaml` or set `RAVEX_ENABLED=1`
- `ravex disable` removes the autoloader; `RAVEX_ENABLED=0` turns it off for a
  single run

## Status and limits

Alpha. Works with plain PyTorch loops, and with anything built on them
(HuggingFace `Trainer`, Lightning, Accelerate) since the hooks are on PyTorch
itself.

Verified: plain loops, gradient accumulation, LR schedulers, AMP loss-scale
state, `num_workers > 0`, DDP, and FSDP. A killed `torchrun` job resumes on
*every* rank with bit-identical losses, sharded or not, and the checkpoint it
leaves behind loads into a plain single-process model afterwards.

### Sharded models

With FSDP each rank holds a slice of every parameter, so `state_dict()` returns
a fragment. Ravex gathers the whole thing, which makes the checkpoint
independent of the topology that produced it: a run sharded over eight GPUs can
be resumed on one.

Two consequences worth knowing:

- Collecting a checkpoint becomes a **collective**. Every rank participates in
  the gather; only rank 0 writes.
- There is **no final checkpoint at exit** for a sharded model. Shutdown is
  where ranks stop being in lockstep, and a gather nobody else joins hangs.
  Losing the last few steps is bounded; a hang is not. Set `checkpoint_every`
  accordingly.

Known limits today:

- **`IterableDataset`**: no index sampler exists, so the stream position cannot
  be replayed. Everything else is still restored.
- **Very large sharded models**: the gather is a full state dict, so rank 0
  needs to hold the model in CPU memory. Per-rank sharded checkpoints — which
  Moonclip already supports — are the answer for models past that point, and
  are not wired up yet.
- **AMP on CUDA, and FSDP1**: tested on CPU only. There is no GPU in the
  development environment, and FSDP1 refuses to initialise without an
  accelerator, so the tests use FSDP2. Both go through the same gather.
- **Your loop's bounds**: a resumed script runs its own `for epoch in
  range(N)` again from the top; it has no idea 3000 steps already happened. Set
  `max_steps` and Ravex ends the run at the right step regardless of how many
  times the process restarted.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The unit suite runs in-process. The parts that only exist across a real process
boundary — the `.pth` autoloader, a resume starting from an empty interpreter,
`torchrun` — live in `integration/` and need Linux:

```bash
docker build -f integration/Dockerfile -t ravex-integration .
docker run --rm ravex-integration
```

## Licence

Apache 2.0. See [LICENSE](LICENSE).
