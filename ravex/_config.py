"""
Configuration for the Ravex runtime.

Resolution order (later wins):

1. built-in defaults
2. ``ravex.yaml`` — looked up at ``RAVEX_CONFIG``, else the first one
   found walking up from the current working directory
3. ``RAVEX_*`` environment variables

Environment variables always win so that a platform (GPU Zero) or a CI job can
override a config file baked into the user's repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_FILENAMES = ("ravex.yaml", "ravex.yml")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def find_config_file() -> Optional[Path]:
    """Locate ``ravex.yaml``.

    ``RAVEX_CONFIG`` takes precedence; otherwise walk up from the cwd, so
    that running ``python train/run.py`` from a subdirectory still finds the
    config at the repository root.
    """
    explicit = os.environ.get("RAVEX_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    try:
        start = Path.cwd()
    except OSError:  # cwd deleted underneath us
        return None

    for directory in (start, *start.parents):
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        # PyYAML is a declared dependency, but Ravex must never be the
        # reason a training run fails to start.
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class StorageConfig:
    """Where checkpoints go.

    ``type`` is ``local``, ``s3`` or ``r2``. ``r2`` is S3-compatible and only
    differs in that a custom endpoint is mandatory.
    """

    type: str = "local"
    path: str = "./checkpoints"
    bucket: Optional[str] = None
    prefix: str = ""
    endpoint: Optional[str] = None
    region: str = "us-east-1"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    path_style: bool = False

    @property
    def is_remote(self) -> bool:
        return self.type in ("s3", "r2")

    def resolve_credentials(self) -> None:
        """Fill missing credentials from the environment.

        Credentials are never expected in ``ravex.yaml`` — that file lives
        in the user's repository.
        """
        if self.access_key is None:
            self.access_key = os.environ.get(
                "RAVEX_S3_ACCESS_KEY"
            ) or os.environ.get("AWS_ACCESS_KEY_ID")
        if self.secret_key is None:
            self.secret_key = os.environ.get(
                "RAVEX_S3_SECRET_KEY"
            ) or os.environ.get("AWS_SECRET_ACCESS_KEY")


@dataclass
class RavexConfig:
    """Full runtime configuration."""

    enabled: bool = True

    # Checkpoint cadence, counted in optimizer steps (not micro-batches, so
    # gradient accumulation is handled for free).
    checkpoint_every: int = 500
    checkpoint_on_exit: bool = True
    resume: bool = True

    # Optional hard stop, in optimizer steps. Without it, a resumed script runs
    # its own loop bounds again from the top and overshoots the intended
    # budget; with it, Ravex ends the run at the right step no matter how many
    # times the process restarted. See docs/configuration.md.
    max_steps: Optional[int] = None

    backend: str = "moonclip"
    storage: StorageConfig = field(default_factory=StorageConfig)

    delta: bool = True
    compression: str = "zstd"
    compression_level: int = 3
    keep_last: int = 5

    # How often each rank sends a copy of its store to a peer on another
    # machine, counted in checkpoints. Only ever used when the storage turns
    # out to be neither remote nor shared — with either of those a copy buys
    # nothing and costs bandwidth. 0 turns it off.
    #
    # Wide on purpose. The average bandwidth is one store per rank divided by
    # this number, so the interval is what keeps replication from becoming
    # backpressure on the training loop. What it costs in exchange is bounded:
    # losing a machine loses at most this many checkpoints of progress.
    replicate_every: int = 10

    # Whether Moonclip keeps the last full snapshot's bytes resident so the
    # next delta can be computed without reading them back from storage.
    #
    # It costs **exactly one extra copy of the saved state** — about +11 GiB
    # for a 1B model with its Adam state — and that copy appears only after
    # the first checkpoint of a run. Which is what makes it the first suspect
    # in GPU-54: `collect` costs half as much at step 2, before any base is
    # retained, as it does from step 4 on. Turning it off is how that
    # hypothesis gets tested, and on a box where host memory is the binding
    # constraint it is also how the memory gets bought back.
    #
    # Defaults to Moonclip's own default rather than to what an experiment
    # would prefer.
    keep_base_in_memory: bool = True

    # Whether Moonclip writes in the background. On, the training loop pays
    # only for the shadow copy and the write drains behind it — which is the
    # entire premise of the handoff being cheap, so this is not a knob to turn
    # off casually.
    #
    # It exists because "off" is the only way to ask whether the *next*
    # collection is competing with the previous write. With it on there is a
    # writer running during every collection except the first, which is exactly
    # the shape GPU-54 measures.
    async_save: bool = True

    # How a sharded (FSDP) model gets written.
    #
    # ``gather``    the whole state is collected on rank 0, which writes it.
    #               The checkpoint is independent of the topology — eight GPUs
    #               in, one out — and it does not scale: 14.2 s and 18.1 GiB of
    #               resident memory on rank 0 for a 1.48B model.
    # ``per_rank``  every rank writes its own shard into its own store. Nothing
    #               is gathered, so nothing is bounded by one rank's memory, and
    #               the checkpoint only resumes at the same world size with the
    #               same sharding.
    #
    # Defaults to ``gather`` because losing the ability to resume on a different
    # number of GPUs is not something to acquire by upgrading.
    sharded_checkpoints: str = "gather"

    # Resume a ``per_rank`` checkpoint onto a different number of ranks,
    # stitching each new shard out of the old ones.
    #
    # Off by default, and the reason is not caution about the arithmetic. A
    # resume that reshards silently is a resume that silently succeeds when the
    # launcher was misconfigured and started 3 ranks where the job wants 4 —
    # the run continues, the loss looks plausible, and nothing says the world
    # shrank. The mismatch is therefore always detected and always logged; only
    # acting on it is opt-in.
    #
    # Preconditions, checked at resume and refused loudly when unmet: a 1-D
    # mesh (FSDP, ``Shard`` and ``Replicate`` only), and every old rank's store
    # readable from here — as itself or as a complete peer copy. See
    # ``docs/how-it-works.md`` for what a resharded resume does *not* promise,
    # which is the data order and the per-rank RNG.
    reshard_on_resume: bool = False

    # Interception toggles — each patch can be disabled independently, which
    # makes bisecting an incompatibility trivial.
    track_dataloaders: bool = True
    track_rng: bool = True

    # Checkpoint on SIGTERM: what a preempted spot instance gets, ~10s before
    # it is killed.
    handle_sigterm: bool = True

    log_file: Optional[str] = None
    log_level: str = "INFO"
    fallback_on_error: bool = True

    framework_auto_detect: bool = True

    # Purely informational, surfaced in checkpoint metadata.
    run_id: Optional[str] = None

    source: Optional[str] = None  # path of the yaml this came from, if any

    #: Values that had to be replaced while loading. Logged by the runtime once
    #: logging exists, so a broken config is visible instead of silent.
    problems: List[str] = field(default_factory=list)

    # ─── loading ────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "RavexConfig":
        config = cls()
        path = find_config_file()
        if path is not None:
            config.source = str(path)
            config._apply_mapping(_read_yaml(path))
        config._apply_env()
        config.storage.resolve_credentials()
        config._normalize()
        return config

    def _apply_mapping(self, data: Dict[str, Any]) -> None:
        if not data:
            return

        storage = data.pop("storage", None)
        if isinstance(storage, dict):
            for key, value in storage.items():
                if hasattr(self.storage, key):
                    setattr(self.storage, key, value)

        frameworks = data.pop("frameworks", None)
        if isinstance(frameworks, dict) and "auto_detect" in frameworks:
            self.framework_auto_detect = _as_bool(
                frameworks["auto_detect"], self.framework_auto_detect
            )

        known = {f.name for f in fields(self)}
        for key, value in data.items():
            if key in known and key not in ("storage", "source", "problems"):
                setattr(self, key, value)

    def _apply_env(self) -> None:
        env = os.environ

        def get(name: str) -> Optional[str]:
            return env.get(f"RAVEX_{name}")

        if (value := get("ENABLED")) is not None:
            self.enabled = _as_bool(value, self.enabled)
        if (value := get("CHECKPOINT_EVERY")) is not None:
            self.checkpoint_every = _as_int(value, self.checkpoint_every)
        if (value := get("CHECKPOINT_ON_EXIT")) is not None:
            self.checkpoint_on_exit = _as_bool(value, self.checkpoint_on_exit)
        if (value := get("RESUME")) is not None:
            self.resume = _as_bool(value, self.resume)
        if (value := get("MAX_STEPS")) is not None:
            self.max_steps = _as_int(value, 0) or None
        if (value := get("BACKEND")) is not None:
            self.backend = value
        if (value := get("DELTA")) is not None:
            self.delta = _as_bool(value, self.delta)
        if (value := get("COMPRESSION")) is not None:
            self.compression = value
        if (value := get("COMPRESSION_LEVEL")) is not None:
            self.compression_level = _as_int(value, self.compression_level)
        if (value := get("KEEP_LAST")) is not None:
            self.keep_last = _as_int(value, self.keep_last)
        if (value := get("REPLICATE_EVERY")) is not None:
            self.replicate_every = _as_int(value, self.replicate_every)
        if (value := get("KEEP_BASE_IN_MEMORY")) is not None:
            self.keep_base_in_memory = _as_bool(value, self.keep_base_in_memory)
        if (value := get("ASYNC_SAVE")) is not None:
            self.async_save = _as_bool(value, self.async_save)
        if (value := get("SHARDED_CHECKPOINTS")) is not None:
            self.sharded_checkpoints = value
        if (value := get("RESHARD_ON_RESUME")) is not None:
            self.reshard_on_resume = _as_bool(value, self.reshard_on_resume)
        if (value := get("TRACK_DATALOADERS")) is not None:
            self.track_dataloaders = _as_bool(value, self.track_dataloaders)
        if (value := get("TRACK_RNG")) is not None:
            self.track_rng = _as_bool(value, self.track_rng)
        if (value := get("HANDLE_SIGTERM")) is not None:
            self.handle_sigterm = _as_bool(value, self.handle_sigterm)
        if (value := get("LOG_FILE")) is not None:
            self.log_file = value or None
        if (value := get("LOG_LEVEL")) is not None:
            self.log_level = value
        if (value := get("FALLBACK_ON_ERROR")) is not None:
            self.fallback_on_error = _as_bool(value, self.fallback_on_error)
        if (value := get("RUN_ID")) is not None:
            self.run_id = value

        # Storage
        if (value := get("STORAGE_TYPE")) is not None:
            self.storage.type = value
        if (value := get("STORAGE_PATH")) is not None:
            self.storage.path = value
        if (value := get("STORAGE_BUCKET")) is not None:
            self.storage.bucket = value
        if (value := get("STORAGE_PREFIX")) is not None:
            self.storage.prefix = value
        if (value := get("STORAGE_ENDPOINT")) is not None:
            self.storage.endpoint = value
        if (value := get("STORAGE_REGION")) is not None:
            self.storage.region = value
        if (value := get("STORAGE_PATH_STYLE")) is not None:
            self.storage.path_style = _as_bool(value, self.storage.path_style)

    def _coerce(self) -> None:
        """Force every field to its declared type, replacing anything unusable.

        A config file is written by hand, or by a template, or by a job runner,
        and it arrives however it arrives. A truncated ``ravex.yaml`` - one
        interrupted write, one bad template - used to yield ``checkpoint_every:
        None``, which crashed the runtime on construction. The autoloader
        swallows that exception by design, so the result was Ravex silently not
        starting: no log file, because logging is configured after the config
        loads, and no checkpoints, on a run that had asked for them.

        Bad values are replaced with defaults and recorded in ``problems``,
        which the runtime logs once it has somewhere to log to.
        """
        defaults = RavexConfig()
        numeric = (
            "checkpoint_every",
            "keep_last",
            "compression_level",
            "replicate_every",
        )
        boolean = (
            "enabled",
            "checkpoint_on_exit",
            "resume",
            "delta",
            "keep_base_in_memory",
            "async_save",
            "reshard_on_resume",
            "track_dataloaders",
            "track_rng",
            "handle_sigterm",
            "fallback_on_error",
            "framework_auto_detect",
        )

        for name in numeric:
            value = getattr(self, name)
            coerced = _as_int(value, getattr(defaults, name))
            if coerced != value:
                self.problems.append(
                    f"{name}={value!r} is not a number; using {coerced}"
                )
            setattr(self, name, coerced)

        for name in boolean:
            value = getattr(self, name)
            if not isinstance(value, bool):
                coerced = _as_bool(value, getattr(defaults, name))
                self.problems.append(
                    f"{name}={value!r} is not true/false; using {coerced}"
                )
                setattr(self, name, coerced)

        if self.max_steps is not None:
            self.max_steps = _as_int(self.max_steps, 0) or None

        for name in ("backend", "compression", "log_level", "sharded_checkpoints"):
            value = getattr(self, name)
            if not isinstance(value, str):
                replacement = getattr(defaults, name)
                self.problems.append(
                    f"{name}={value!r} is not a name; using {replacement!r}"
                )
                setattr(self, name, replacement)

        if not isinstance(self.storage.path, str):
            self.problems.append(
                f"storage.path={self.storage.path!r} is not a path; using "
                f"{defaults.storage.path!r}"
            )
            self.storage.path = defaults.storage.path

    def _normalize(self) -> None:
        self._coerce()
        self.backend = str(self.backend).strip().lower()
        self.storage.type = str(self.storage.type).strip().lower()
        self.log_level = str(self.log_level).strip().upper()

        self.sharded_checkpoints = str(self.sharded_checkpoints).strip().lower()
        if self.sharded_checkpoints not in ("gather", "per_rank"):
            # Not a typo to guess at: "per_rank" gives up resuming at a
            # different world size, so an unrecognised value falls back to the
            # mode that keeps every option open.
            self.problems.append(
                f"sharded_checkpoints={self.sharded_checkpoints!r} is not "
                "'gather' or 'per_rank'; using 'gather'"
            )
            self.sharded_checkpoints = "gather"

        if self.checkpoint_every < 1:
            self.checkpoint_every = 1
        if self.keep_last < 1:
            self.keep_last = 1
        if self.replicate_every < 0:
            self.replicate_every = 0
        if str(self.compression).strip().lower() in ("none", "off", ""):
            self.compression_level = 0

        # A remote store without a bucket is a misconfiguration; degrade to
        # local rather than failing the run.
        if self.storage.is_remote and not self.storage.bucket:
            self.storage.type = "local"

        if self.run_id and not self.storage.prefix:
            self.storage.prefix = self.run_id

    def describe(self) -> str:
        target = (
            f"{self.storage.type}://{self.storage.bucket}/{self.storage.prefix}"
            if self.storage.is_remote
            else self.storage.path
        )
        return (
            f"backend={self.backend} storage={target} "
            f"every={self.checkpoint_every} keep_last={self.keep_last} "
            f"config={self.source or '<defaults>'}"
        )
