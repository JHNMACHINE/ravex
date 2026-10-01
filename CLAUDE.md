# CLAUDE.md

Ravex: a PyTorch training runtime, Python with a Rust core (the reshard
planner, `src/`). The README's Development section is the setup; this file is
what it doesn't say.

## Testing

- **Use the repository's own venv**: `.venv/Scripts/python -m pytest tests -q`
  (Windows) - never a bare `pytest`, which may be another interpreter without
  Moonclip, and then the suite silently exercises only the `torch_save`
  fallback, not the backend that ships.
- **The suite needs the compiled core in the source tree** (`ravex/_core*`).
  Without it the reshard tests fail at collection. After a **version bump**
  rebuild it even with no Rust change: `test_version_is_single.py` compares the
  version baked into `_core` with `ravex.__version__`.
- Rust gate, faster and interpreter-free: `cargo test --no-default-features`,
  then clippy **twice**, with and without default features - as `checks.yml`
  does. `--no-default-features` compiles `src/python.rs` out, so the binding
  layer is only linted by the second run.
- `tests/` and `integration/` both have a `conftest.py`: run them in separate
  pytest invocations, or the module names collide.
- `integration/` needs Linux (SIGKILL, `torchrun`): build
  `integration/Dockerfile`. Run containers that simulate a preemption with
  `--init` - SIGKILL sent by PID 1 to itself is a no-op, and the bench goes
  green having proved nothing. The image has no procps: read `/proc` rather
  than trusting `$(pgrep ... || echo 0)`.
- Without `RAVEX_LOG_FILE` only WARNING and above reach stderr, so topology
  announcements are invisible in a bench's output.

## Multi-process tests

The `moonclip-backend` job in `ci.yml` runs the suite under `pytest -n 4
--dist loadfile`, so a test shares the box with other workers' stores and
ports.

- **Never probe a free port and hand it over.** Another worker can take it
  before the listen (EADDRINUSE), or a rank ends up talking to someone else's
  store. Listen on port 0 and read the port back - `rendezvous.serve(host, 0)`
  then `.port`, or a `TCPStore` the test's own process holds (GPU-175).
- **Don't assert that a race goes the bad way.** A test that needs torch to
  fail is red the day torch wins. Pin the mechanism of the fix instead.

## Things that cost a session to find

- **Windows CPU wheels of torch have no libuv.** `torchrun`'s c10d rendezvous
  cannot run on Windows, and `USE_LIBUV=0` is not honoured. A hand-built
  `TCPStore(..., use_libuv=False)` works (`rendezvous._tcp_store` falls back to
  it); anything elastic via `torchrun` goes in the integration container.
- **Deadlines on a socket whose fd goes into Rust** use
  `_dist.exchange.set_deadline` (`SO_RCVTIMEO`/`SO_SNDTIMEO`), never
  `settimeout()`: that makes the fd non-blocking, and Rust's first read
  returns `WouldBlock` having moved nothing.
- **Never open Moonclip's internal files** (`manifest.json` and the like)
  while a manager is alive. On Windows a reader blocks Moonclip's rename and
  the checkpoint being written is lost. Ask the API: `describe()`.
- `core.autocrlf` checkouts on Windows: edit by script with the line ending
  the file already has, or the whole file shows as changed.
