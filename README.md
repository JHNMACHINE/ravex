# Ravex

Checkpoint and resume for PyTorch training, from one decorator on the function
that trains. Inside that function nothing changes: Ravex notices the model, the
optimizer, the LR scheduler, the AMP scaler and the dataloader as they are
built, counts every `optimizer.step()`, and writes a checkpoint on the cadence
you configured. If the process dies, running the same command again picks up
where it stopped.

## Install

```bash
pip install ravex
```

Ravex is pure Python and needs Python 3.11 or later. It does not depend on
torch, so that it attaches to whatever build you already have (CUDA, ROCm, CPU)
instead of pulling another one in.

Its default checkpoint engine is [Moonclip](https://github.com/JHNMACHINE/moonclip),
installed with `pip install "ravex[moonclip]"`. Moonclip publishes wheels only
for Linux x86_64; elsewhere it has to be built from source, and where it is not
installed Ravex falls back to `torch_save` on its own.

## Use

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

A resumed run continues rather than restarting from a good place, because
Ravex restores everything around the weights too: optimizer state, LR
scheduler, AMP loss scale, the torch, CUDA, Python and NumPy RNGs, and the
position in the dataset. The test suite holds it to the strong form of that: a
run killed at step 20 and resumed produces the same losses, step by step, and
the same final weights as the run that was never interrupted.

`preemption_handler=True` catches SIGTERM and tries to write a final checkpoint
before the process is killed. Any other keyword argument to `train_loop` is a
configuration override, and an unknown name raises `TypeError` rather than
being ignored.

The decorator is the only way in. Up to 0.0.5 a `.pth` file in site-packages
attached Ravex to any process with a `ravex.yaml` above it; it was removed so
that nothing is ever checkpointed by code that does not say so. Importing
Ravex does nothing until a decorated function is called, and its patches on
PyTorch come off when that function returns or raises.

When "noticed on its own" is a guess, such as two models of which only one
trains, or an optimizer built before the function was entered, name it with
`ravex.track(model, optimizer)`. The one moment a checkpoint is consistent,
the top of an iteration, Ravex takes from the `DataLoader` iterator or, in a
loop without one, from the first forward of the trained model after each
step; `ravex.batch_boundary()` hands it over by hand where neither is seen.

## Configuration

`ravex.yaml`, at or above the working directory:

```yaml
checkpoint_every: 500        # optimizer steps between checkpoints (default 100)
backend: moonclip            # moonclip | torch_save
storage:
  type: local                # local | s3 | r2
  path: ./checkpoints
keep_last: 5
max_steps: null              # stop at this step however many times the process restarted
sharded_checkpoints: gather  # gather | per_rank, for FSDP
```

Every option is also read from a `RAVEX_*` environment variable, which wins
over the file, and the keyword arguments to `train_loop` win over both. A
resume is the default (`resume: true`); `RAVEX_ENABLED=0` turns Ravex off for
one run without touching the code.

**Storage.** Ravex decides when to save and what, and puts it back into the
live objects; Moonclip decides how the bytes are written and where, bucket
included. So `storage` says where the bucket is, and the credentials are never
read by Ravex or from the config file: Moonclip reads them from
`MOONCLIP_S3_ACCESS_KEY` / `MOONCLIP_S3_SECRET_KEY` or the usual `AWS_*` pair
([Moonclip's README](https://github.com/JHNMACHINE/moonclip#s3-backend)).

**More than one GPU or machine.** DDP and FSDP resume on every rank. With FSDP,
`gather` writes one checkpoint that loads at any world size, bounded by what
rank 0 can hold; `per_rank` has each rank write its own shard, which is much
cheaper and resumes only at the same world size (`reshard_on_resume` handles a
different one after a restart). Across machines without a bucket or a shared
filesystem, `replicate_every` copies each rank's store to a peer on another
machine, so losing one machine costs at most that many checkpoints; with a
bucket, a rank that comes up with an empty disk pulls its store back. Ravex
also trains across slow links between distant nodes with an outer loop that
exchanges parameter differences every `outer_inner_steps` instead of gradients
every step.

The full option list, with the measurements behind the defaults:
[docs/configuration.md](docs/configuration.md). How the patches, the step
count and the resume work: [docs/how-it-works.md](docs/how-it-works.md).

## Limits

- Under HuggingFace `Trainer` and Lightning the state comes back exactly, but
  with shuffling and dropout on, the resumed run sees a different random draw,
  because the framework drives the dataloader and the RNG on its own schedule.
  Plain loops, DDP and FSDP replay exactly.
- An `IterableDataset` has no index sampler, so its position cannot be
  replayed; everything else is still restored.
- A sharded model gets no final checkpoint at exit, since collecting one is a
  collective and ranks stop being in lockstep at shutdown; the newest
  checkpoint is the last periodic one.
- A failed checkpoint is lost, logged, and tried again at the next one;
  a failure inside Ravex never stops the training (`fallback_on_error`).

What changed between versions is in the [CHANGELOG](CHANGELOG.md).

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit
```

`tests/unit` runs in one process on any OS. `tests/distributed` and
`integration/` start several processes, use `torchrun` or kill a process with
SIGKILL, and need Linux. The integration tests run in their own image, with
`--init` so that the SIGKILL they send is not a no-op:

```bash
docker build -f integration/Dockerfile -t ravex-integration .
docker run --rm --init ravex-integration
```

## Licence

Apache-2.0 — [GPU Zero](https://gpuzero.dev)
