"""The Ravex runtime.

One instance per process. It owns the registry, the checkpoint backend and the
resume logic, and it is what the PyTorch patches call into.

Two invariants shape everything here:

*Never break the training run.* Every entry point is wrapped; on an unexpected
error the runtime disables itself (``fallback_on_error``) and the user's code
carries on as if Ravex were not installed.

*Never write to stdout.* Training output belongs to the user. Logs go to the
configured file, or to stderr at WARNING and above when no file is set —
silently swallowing a checkpoint failure would be worse than three lines of
stderr.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Optional

from ravex._backends import get_backend
from ravex._config import RavexConfig
from ravex._distributed import get_rank, get_world_size, is_main_process
from ravex._patches import install_all_patches
from ravex._registry import ObjectRegistry
from ravex._resume import ResumeManager

logger = logging.getLogger("ravex")

_LOG_FORMAT = "%(asctime)s [ravex] %(levelname)s %(message)s"


class RavexRuntime:
    """Singleton runtime. Build it through :func:`get_runtime`."""

    def __init__(self, config: Optional[RavexConfig] = None):
        self.config = config or RavexConfig.load()
        self.registry = ObjectRegistry()

        self._enabled = self.config.enabled
        self._patches = None
        self._backend = None
        self._resume_manager: Optional[ResumeManager] = None
        self._leader_optimizer_id: Optional[int] = None
        self._checkpoint_due = False
        self._last_saved_step: Optional[int] = None
        self._shutdown_done = False
        self._lock = threading.RLock()
        self._framework = "unknown"

        self._setup_logging()

    # ─── lifecycle ──────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        logger.handlers.clear()
        logger.propagate = False  # the user's root logger stays untouched

        level = getattr(logging, self.config.log_level, logging.INFO)
        if self.config.log_file:
            try:
                handler: logging.Handler = logging.FileHandler(
                    self.config.log_file, encoding="utf-8"
                )
                handler.setLevel(level)
            except OSError:
                handler = logging.StreamHandler(sys.stderr)
                handler.setLevel(logging.WARNING)
        else:
            # No log file: stay out of the way, but do not hide failures.
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(max(level, logging.WARNING))

        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)

    def activate(self) -> None:
        """Install the PyTorch patches and get ready to checkpoint."""
        if not self._enabled:
            logger.debug("Ravex is disabled by configuration")
            return

        try:
            from ravex._frameworks import detect_framework

            self._framework = detect_framework()
            self._backend = get_backend(self.config)
            self._resume_manager = ResumeManager(self._backend, self.registry)
            self._patches = install_all_patches(self.registry, self)
            self._install_signal_handler()

            logger.info(
                "Ravex active - %s framework=%s rank=%d/%d",
                self.config.describe(),
                self._framework,
                get_rank(),
                get_world_size(),
            )
        except Exception as exc:
            logger.error("Activation failed: %s", exc, exc_info=True)
            self._disable("activation failed")

    def _disable(self, reason: str) -> None:
        if not self._enabled:
            return
        self._enabled = False
        logger.error("Ravex disabled (%s) - training continues unaffected", reason)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self):
        return self._backend

    @property
    def step(self) -> int:
        return self.registry.step_count

    # ─── hooks called from the patches ──────────────────────────────

    def on_step(self, optimizer: Any) -> None:
        """Called after every ``optimizer.step()``.

        The counter advances in optimizer steps, not micro-batches, so gradient
        accumulation needs no special handling: ``step()`` fires once per
        accumulation cycle by construction.

        With several optimizers (a GAN's generator and discriminator, say) only
        the first one registered drives the counter, so one training iteration
        stays one step.
        """
        if not self._enabled:
            return

        if self._leader_optimizer_id is None:
            self._leader_optimizer_id = id(optimizer)
        elif id(optimizer) != self._leader_optimizer_id:
            return

        # Last chance to resume: reached only when the run has no DataLoader,
        # since DataLoader.__iter__ fires earlier.
        if self._resume_manager is not None and not self._resume_manager.attempted:
            self._try_resume()

        self.registry.step_count += 1
        if self.registry.step_count % self.config.checkpoint_every == 0:
            # Not here: this is the middle of the iteration, before the LR
            # scheduler has stepped. Flag it and collect at the top of the next
            # one — see _BatchBoundaryIterator.
            self._checkpoint_due = True
            if not self.registry.dataloaders:
                # No dataloader to give us that boundary (a loop over
                # pre-batched tensors). Mid-iteration is then the best
                # available point.
                self._checkpoint_due = False
                self.checkpoint()

    def on_batch_boundary(self) -> None:
        """Called at the top of each training iteration, before the batch."""
        if not self._enabled or not self._checkpoint_due:
            return
        self._checkpoint_due = False
        self.checkpoint()

    def should_stop(self) -> bool:
        """Whether the configured step budget has been spent.

        A resumed script re-runs its own loop bounds from the top — it has no
        idea 3000 of its 5000 steps already happened in a previous process. So
        when ``max_steps`` is set, the dataloaders start returning empty:
        remaining epochs fall through instantly and the script exits on its own
        terms, without Ravex raising anything into user code.
        """
        if not self._enabled or self.config.max_steps is None:
            return False
        return self.registry.step_count >= self.config.max_steps

    def on_dataloader_iter(self, _loader: Any) -> None:
        """Called on ``iter(dataloader)``.

        This is the moment to resume: the model, the optimizer and the loader
        all exist, and no batch has been drawn yet — so the restored dataset
        position and RNG state are applied before they are used.
        """
        if not self._enabled or self._resume_manager is None:
            return
        if self._resume_manager.attempted:
            return
        if self.registry.is_empty():
            return  # nothing to restore into yet
        self._try_resume(defer_rng=True)

    def after_dataloader_iter(self) -> None:
        """Called once the loader's iterator exists.

        Constructing it draws a worker base seed from the global RNG, so the
        restored RNG state is applied here rather than in
        :meth:`on_dataloader_iter` — otherwise the first resumed step would run
        one draw ahead of the run it is meant to continue.
        """
        if self._enabled:
            self.registry.apply_pending_rng()

    def _try_resume(self, defer_rng: bool = False) -> None:
        if not self.config.resume:
            if self._resume_manager is not None:
                self._resume_manager.attempted = True
            return
        try:
            self._resume_manager.try_resume(defer_rng=defer_rng)
        except Exception as exc:
            logger.warning("Resume failed (%s) - starting from scratch", exc)
            if self._resume_manager is not None:
                self._resume_manager.attempted = True

    # ─── checkpointing ──────────────────────────────────────────────

    def checkpoint(self, final: bool = False) -> bool:
        """Collect and hand off a checkpoint. Returns True if one was written.

        Collection runs on the training thread on purpose: the state has to be
        consistent with the step that just finished. The backend copies it and
        writes in the background, so what the loop actually pays for is the
        copy, not the I/O.
        """
        if not self._enabled or self._backend is None:
            return False
        if not is_main_process():
            return False

        step = self.registry.step_count
        if self._last_saved_step == step and not final:
            return False

        started = time.perf_counter()
        try:
            state = self.registry.collect_state(track_rng=self.config.track_rng)
            metadata = {
                "step": str(step),
                "framework": self._framework,
                "world_size": str(get_world_size()),
            }
            if self.config.run_id:
                metadata["run_id"] = self.config.run_id
            if final:
                metadata["final"] = "true"

            self._backend.save(step, state, metadata)
            self._last_saved_step = step
        except Exception as exc:
            logger.error("Checkpoint at step %d failed: %s", step, exc, exc_info=True)
            if self.config.fallback_on_error:
                self._disable("checkpoint failed")
            return False

        logger.info(
            "Checkpoint at step %d handed off in %.3fs",
            step,
            time.perf_counter() - started,
        )
        return True

    def flush(self) -> None:
        if self._backend is not None:
            try:
                self._backend.flush()
            except Exception as exc:
                logger.warning("Flush failed: %s", exc)

    # ─── shutdown ───────────────────────────────────────────────────

    def _install_signal_handler(self) -> None:
        """Checkpoint on SIGTERM — the signal a preempted spot instance gets.

        Only installed when nothing else has claimed SIGTERM and only on the
        main thread; stealing the user's handler would be worse than missing
        the last few steps.
        """
        if not self.config.handle_sigterm or os.name == "nt":
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
                logger.debug("SIGTERM already handled elsewhere; not installing")
                return

            def handler(signum, frame):
                logger.warning("SIGTERM received - saving before exit")
                self.shutdown()
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

            signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
            logger.debug("Could not install SIGTERM handler: %s", exc)

    def shutdown(self) -> None:
        """Final checkpoint plus flush. Safe to call more than once."""
        with self._lock:
            if self._shutdown_done or not self._enabled:
                return
            self._shutdown_done = True

        try:
            if (
                self.config.checkpoint_on_exit
                and self.registry.step_count > 0
                and self.registry.step_count != self._last_saved_step
            ):
                self.checkpoint(final=True)
            if self._backend is not None:
                self._backend.close()
            logger.info("Ravex shutdown complete at step %d", self.registry.step_count)
        except Exception as exc:
            logger.warning("Shutdown error: %s", exc)


# ─── process-wide instance ──────────────────────────────────────────

_runtime: Optional[RavexRuntime] = None
_runtime_lock = threading.Lock()


def get_runtime(create: bool = True) -> Optional[RavexRuntime]:
    """Return the process runtime, building it on first use."""
    global _runtime
    if _runtime is None and create:
        with _runtime_lock:
            if _runtime is None:
                _runtime = RavexRuntime()
    return _runtime


def reset_runtime() -> None:
    """Tear the runtime down. For tests; not part of the public API."""
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            if _runtime._patches is not None:
                _runtime._patches.uninstall()
            try:
                if _runtime._backend is not None:
                    _runtime._backend.close()
            except Exception:
                pass
            import torch

            if hasattr(torch.nn.Module, "_ravex_patched"):
                delattr(torch.nn.Module, "_ravex_patched")
        _runtime = None
