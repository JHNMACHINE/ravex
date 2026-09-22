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
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


#: How the outer loop weights each node's contribution. Mirrors
#: ``ravex._dist.outer.COMBINE_MODES``, and duplicated for the same reason
#: ``_SAVE_DTYPES`` is: this has to be checkable at config load, which happens
#: before anything under ``_dist`` is imported and long before a round.
_OUTER_COMBINE_MODES = ("mean", "normalized", "step_weighted")


#: Every ``save_dtype`` target Moonclip accepts, in every spelling it accepts
#: it. Duplicated here on purpose: Moonclip reports a bad target when the
#: manager is *constructed*, which under Ravex is mid-run and inside the
#: ``except`` that falls back to ``torch.save``. Checking here puts a typo in
#: ``problems`` at config load instead, next to every other bad value.
#:
#: Integers, ``bool`` and complex are absent because they are not cast
#: targets: they travel through unchanged whatever this is set to.
_SAVE_DTYPES = frozenset(
    {
        "none",
        "bf16",
        "bfloat16",
        "fp16",
        "float16",
        "fp32",
        "float32",
        "fp64",
        "float64",
        "double",
        "fp8",
        "float8",
        "fp8_e4m3",
        "float8_e4m3fn",
        "fp8_e5m2",
        "float8_e5m2",
    }
)

#: Component name → the tensor-name globs it covers.
#:
#: Moonclip matches patterns and knows nothing about optimizers, which is the
#: right division: the structure lives in the names, and Ravex is the layer
#: that chose them. So Ravex is also the layer that should spell out what a
#: component *is* — and there is a real reason not to leave that to the user.
#:
#: **A component lives in two places, not one.** An unsharded optimizer is
#: written under ``ravex/optimizers/…``; the same optimizer under FSDP is
#: written under ``ravex/sharded/<key>/optimizer/…``, because sharded state
#: goes through its own collection path. Someone writing patterns by hand
#: would reach for ``ravex/optimizers/*``, get it right on a single GPU, and
#: silently cast nothing on the sharded run that motivated the setting — which
#: is precisely the case the measurement came from. Naming the component
#: covers both.
_SAVE_DTYPE_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "model": ("ravex/models/*", "ravex/sharded/*/model/*"),
    "optimizer": ("ravex/optimizers/*", "ravex/sharded/*/optimizer/*"),
    "scheduler": ("ravex/schedulers/*",),
    "scaler": ("ravex/scalers/*",),
    "dataloader": ("ravex/dataloaders/*",),
}


def _save_dtype_from_env(value: str) -> Union[str, Dict[str, str]]:
    """Parse ``RAVEX_SAVE_DTYPE``.

    Two forms, distinguished by a colon:

        RAVEX_SAVE_DTYPE=bf16
        RAVEX_SAVE_DTYPE=optimizer:bf16,model:none

    The second is comma-separated and **ordered**, because the rules are: the
    first match wins, and an environment variable is a list of characters, so
    the order is right there for free.

    Nothing is validated here. Whatever comes out goes through
    :meth:`RavexConfig._normalize_save_dtype` like a value from the YAML,
    which is the one place that decides what is usable and records what was
    not — two checks would eventually disagree.
    """
    text = value.strip()
    if ":" not in text:
        return text

    rules: Dict[str, str] = {}
    for clause in text.split(","):
        key, _, dtype = clause.partition(":")
        key = key.strip()
        if key:
            rules[key] = dtype.strip()
    return rules


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

    # What precision each part of the checkpoint is stored at.
    #
    #   save_dtype: bf16                    # everything
    #   save_dtype: {optimizer: bf16}       # only the optimizer state
    #   save_dtype: {model: none, "*": bf16}  # everything except the weights
    #
    # Keys are component names — ``model``, ``optimizer``, ``scheduler``,
    # ``scaler``, ``dataloader`` — or raw Moonclip globs over the tensor name
    # for anything they do not cover. **The first matching rule wins**, in the
    # order written, which is what makes the third line above express an
    # exception rather than a contradiction.
    #
    # Why it is worth setting. Measured 2026-08-18 on 8× RTX 5060 Ti, a 1.5B
    # model under FSDP2, per rank: the weights are about a third of the bytes
    # and delta well, −70%. `exp_avg` and `exp_avg_sq` are the other two
    # thirds and delta essentially not at all, −1.1% and −4.0% — between two
    # steps their XOR is high-entropy, because with β₁ = 0.9 a tenth of the
    # value is replaced by fresh gradient every step and that moves nearly
    # every mantissa bit. So **85% of what gets written is the part that does
    # not compress**, and it is also the part that tolerates the least
    # precision: `exp_avg_sq` reaches Adam through `sqrt(v)`, which halves the
    # relative error, which is why 8-bit optimizers are ordinary practice.
    # ``{optimizer: bf16}`` halves 85% of the volume and leaves the model
    # exactly as it was.
    #
    # **Defaults to off, and stays off on upgrade.** Turning it on changes the
    # numbers a resumed run gets back, and a library that did that to an
    # in-flight run because someone bumped a version would be wrong to. The
    # backend logs one line per run pointing at this setting when it is unset,
    # which is how it stays discoverable without being imposed.
    #
    # Not every dtype is a target: integers, ``bool`` and complex tensors are
    # transported unchanged whatever this says. Casting weights to ``int8``
    # would be quantization, which needs a scale and a zero-point that a
    # checkpoint entry has nowhere to keep, so the name is refused rather than
    # accepted into something that produces wrong numbers quietly.
    save_dtype: Optional[Union[str, Dict[str, str]]] = None

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

    #: How the replica bytes travel between ranks: ``"sockets"``,
    #: ``"collectives"``, or ``"auto"``.
    #:
    #: Sockets are Ravex's own TCP connections between neighbouring ranks,
    #: framed and driven by the Rust core, and ``auto`` prefers them whenever
    #: the rendezvous store is reachable and every peer's address can be
    #: resolved — which inside a torchrun job it usually is. Measured on one
    #: box, 2 GiB in 8 files: 1611 MB/s against the collectives' 809 at a
    #: 64 MiB chunk, and rising with the chunk size where the collective path
    #: falls (GPU-109).
    #:
    #: ``"collectives"`` is the road back, and it is not deprecated. The
    #: collective path needs no rank to have an address anyone else can reach,
    #: so on a cluster where the ranks cannot open connections to each other it
    #: is the one that works. ``auto`` falls back to it on its own when a link
    #: cannot be built, and says so once.
    replication_transport: str = "auto"

    #: Where the ranks agree about small things: ``"store"``,
    #: ``"collectives"``, or ``"auto"``.
    #:
    #: The questions are tiny — did every rank succeed, what is the newest step
    #: everyone holds — and the answer is a scalar. ``auto`` asks them on the
    #: rendezvous store when there is one. Measured at 4 ranks:
    #: `all_ranks_agree` costs 1544 µs over the collectives and 849 µs here
    #: (GPU-111, `bench/agreement_cost.py`).
    #:
    #: 1.8x, and the reason it is not more is worth knowing: a gather has to
    #: wait for the slowest rank whatever carries it. The larger part of the
    #: change is not the microseconds — it is that a rank going silent produces
    #: an answer here and a group-wide timeout there.
    #:
    #: ``"collectives"`` is the road back, and the reason to keep it is not
    #: performance: a store round decides on its own when a rank goes silent,
    #: and a job that would rather fail together than proceed without one rank
    #: wants the collective's behaviour, not this one's.
    agreement_transport: str = "auto"

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

    # Restore from a checkpoint another framework wrote — a DeepSpeed ZeRO
    # directory, or a torch distributed checkpoint (which is also what
    # Megatron-core writes) — by converting it before the ordinary resume.
    #
    # Off by default for the same reason `reshard_on_resume` is, and it is the
    # stronger case of the two: a foreign checkpoint found where Ravex's own
    # store belongs usually means a path was pointed somewhere unintended, and
    # converting it silently would turn that into a run that trains on someone
    # else's weights without ever saying so. So the detection is unconditional
    # and always logged, and only acting on it is opt-in.
    convert_foreign: bool = False

    # Interception toggles — each patch can be disabled independently, which
    # makes bisecting an incompatibility trivial.
    track_dataloaders: bool = True
    track_rng: bool = True

    # Checkpoint on SIGTERM: what a preempted spot instance gets, ~10s before
    # it is killed.
    handle_sigterm: bool = True

    # The coordinated emergency path for sharded models: on SIGTERM, wait for
    # every rank to notice before attempting the collective save that FSDP
    # checkpointing needs. Independent of `handle_sigterm` — that one governs
    # whether Ravex catches the signal at all; this one governs only the
    # cross-rank coordination, and is a no-op for a plain-replicated (DDP) job
    # or a job confined to one machine, where nothing needs coordinating. See
    # GPU-92 and docs/configuration.md.
    emergency_coordination: bool = True

    #: How often, in optimizer steps, every rank checks whether any rank has
    #: raised the flag. `1` — every step — because the measured local handoff
    #: (10.6s on 8x RTX 5060 Ti, see CHANGELOG) already consumes essentially
    #: the whole ~10s SIGTERM budget on its own, leaving no slack to spend on
    #: detection latency. Not yet validated against the actual per-step cost
    #: of the detection collective on real cross-machine networking (measured
    #: at 7 MB/s on RunPod, overlay-dependent on Vast.ai) — raise this if the
    #: two-machine bench (`integration/two-machines/`) shows it taxing steady
    #: state training more than the emergency path is worth.
    emergency_check_every: int = 1

    #: Timeout, in seconds, for the dedicated group the detection collective
    #: runs on. Deliberately its own group with its own short timeout — see
    #: `ravex._dist.collectives.emergency_group` — so a stuck detection round fails
    #: fast without ever touching the timeout the main process group uses for
    #: ordinary gradient synchronization, which this project's own two-machine
    #: kit has already shown to be flaky enough that shortening it globally
    #: would kill otherwise-healthy runs.
    emergency_timeout: int = 20

    #: How the ranks learn that one of them was preempted: ``"store"``,
    #: ``"collectives"``, or ``"auto"``.
    #:
    #: ``"collectives"`` is what this channel was built as: an
    #: ``all_reduce(MAX)`` of one int32, every step, on the short-timeout
    #: group. The agreement it produces is perfect — it is the same collective
    #: on every rank — and it costs a collective on every step of a run that
    #: is almost never being preempted.
    #:
    #: ``auto`` takes the store where there is a rendezvous: a preempted rank
    #: writes one key naming **the step everybody saves at**, and the ordinary
    #: step becomes a ``check`` on a key that is not there. 83 µs at 4 ranks
    #: against 376, and — the number that decides it — 92 µs against 5.8 ms
    #: when one rank is 5 ms late, because a gather pays the straggler and the
    #: absence of a key does not (GPU-111, `bench/agreement_cost.py`).
    #:
    #: The collective does not disappear on that road, it moves: it is posted
    #: **once**, at the announced step, where it is what proves every rank
    #: arrived before any of them enters a sharded save the others would never
    #: join. What the store removes is the per-step cost, not the agreement.
    emergency_transport: str = "auto"

    # ─── the outer loop, GPU-113 ────────────────────────────────────────
    #
    #: Train across nodes that communicate once every `outer_inner_steps`
    #: rather than every step, exchanging parameter deltas instead of
    #: gradients. See `ravex._dist.outer` for what that is and why a 7 MB/s
    #: link leaves no alternative.
    #:
    #: **Off, and it is the one thing here that never turns itself on.** The
    #: rest of Ravex activates on its own because the worst it does is write a
    #: checkpoint. This changes what the run *trains*: the weights get averaged
    #: with other machines'. An inherited `ravex.yaml` switching that on
    #: quietly would be a run doing something other than what its own code
    #: says, and no log line makes up for that afterwards.
    outer_loop: bool = False

    #: Local optimizer steps per round. The whole saving over synchronous data
    #: parallelism is this number, so it wants to be large — hundreds.
    #:
    #: **Measured** (`bench/outer_convergence.py`, 2026-09-07, a byte-level
    #: transformer on two contiguous shards, 2048 local steps per node). Held
    #: out loss: H=1 **2.69**, H=8 1.51, H=64 **1.45**, H=512 1.78, against
    #: 1.55 for one node given the same wall clock and 1.38 for one node given
    #: the same tokens. Two things worth carrying:
    #:
    #: *Small H is not a safe direction.* H=1 is the worst arm by far, not the
    #: closest thing to synchronous training. An outer step per inner step
    #: compounds a Nesterov buffer at `outer_lr` thousands of times and
    #: overwrites the inner optimizer's progress each time; the outer defaults
    #: are calibrated for H in the hundreds and there is no small-H limit where
    #: this becomes ordinary data parallelism.
    #:
    #: *The ceiling that bench can see is on rounds, not on H.* At equal tokens
    #: a larger H is fewer exchanges, so H=512 there is four of them in the
    #: whole run — which is what degraded, and a long run at H=512 gets
    #: hundreds. So 500 stands: nothing measured argues it down, and what
    #: should actually set it is the link (`bench/round_link_cost.py` reports
    #: the H at which the network is a quarter of the round).
    outer_inner_steps: int = 500

    #: Close the round on the clock instead of, or as well as, on the count.
    #: **This is the one that makes heterogeneous nodes work**: every node
    #: stops at the same moment having done as many steps as it could, and the
    #: step counts differing becomes the normal case rather than a straggler.
    #: 0 disables it. Whichever of the two comes first closes the round.
    outer_round_seconds: float = 0.0

    #: The outer optimizer. DiLoCo's published values; `outer_lr` is not small
    #: on purpose — the step is meant to travel most of the way to the average
    #: of where the nodes went, not to nudge toward it.
    outer_lr: float = 0.7
    outer_momentum: float = 0.9

    #: How contributions are weighted: `mean`, `normalized` or `step_weighted`.
    #: The default is derivable and was then measured — see
    #: `ravex._dist.outer.combine`, where `step_weighted` costs 45% more loss
    #: once the nodes' data actually differs.
    outer_combine: str = "mean"

    #: Seconds a node waits for its peers' reports before closing the round
    #: without them. **A deadline, not a timeout**: running out of it is the
    #: answer rather than a failure, and it is what makes a dead node an
    #: absence. Sized for the link — a round moves one model's worth of bytes.
    outer_deadline: int = 900

    #: Cast the delta before it goes on the wire: `bf16` is 2.56x smaller,
    #: `fp8` 4.84x (measured in `ravex._dist.report`). `none` turns it off.
    #:
    #: **Measured, and still off.** `bench/outer_convergence.py`, 2026-09-07:
    #: at every H tried, the held-out loss with bf16 and with fp8 sits inside
    #: ±0.005 of no cast at all — less than the spread between neighbouring
    #: arms, and fp8 comes out marginally *ahead* at one of them, which is how
    #: you know it is noise. The worry that a delta truncated the same way
    #: every round accumulates its error instead of averaging it out does not
    #: show either: 256 rounds of it at H=8, column flat. On the link this is
    #: for (`bench/round_link_cost.py`, 7 MB/s, 200 ms round trip) it took the
    #: report from 15.8 MB to 11.7 MB and the round's network from 3.28 s to
    #: 2.61 s — **a fifth off every round** — for no measurable change in
    #: publish time.
    #:
    #: So the reason it is still `None` is no longer "unmeasured". It is that
    #: **turning it on is what surfaced the seed-round defect** — the source
    #: node was keeping its own uncast parameters while every peer adopted the
    #: cast ones, and nothing raised — and that the evidence is one 0.48M
    #: model on one box, where a release that changes what a run puts on the
    #: wire wants a two-machine run behind it. Set it, and take the fifth: the
    #: measurement says it is there. It defaults off because the default is a
    #: promise made to people who did not read this comment.
    outer_save_dtype: Optional[str] = None

    #: Where round reports are staged. Defaults to `rounds/` beside the
    #: checkpoint store. Kept apart from the checkpoints deliberately: these
    #: are in-flight working copies with their own retention, and mixing them
    #: into the store a resume reads from is how a round report becomes a
    #: checkpoint nobody meant to keep.
    outer_root: Optional[str] = None

    #: The address of a ``ravex rendezvous`` server, ``host:port`` (GPU-129).
    #: Set, and the outer loop takes its store, its node number and its peers
    #: from there instead of from ``init_process_group``: the nodes are plain
    #: ``python train.py``, need not all be there at the start, and do not lose
    #: the store when one of them dies. Unset, it is torchrun as before. Also
    #: ``RAVEX_RENDEZVOUS``. Read `ravex._dist.rendezvous` for what it does not
    #: do — the server has no authentication.
    outer_rendezvous: Optional[str] = None

    #: The job's name on that server, which keeps several jobs apart on one.
    #: Every node of a run gives the same one. Also ``RAVEX_OUTER_JOB``.
    outer_job: str = "default"

    #: How many nodes start the run together. The first this many to reach the
    #: rendezvous wait for each other and take the starting parameters from
    #: the first; every node after them joins the run in progress (GPU-121).
    #: Also ``RAVEX_OUTER_MIN_NODES``.
    outer_min_nodes: int = 2

    log_file: Optional[str] = None
    log_level: str = "INFO"
    fallback_on_error: bool = True

    framework_auto_detect: bool = True

    # Purely informational, surfaced in checkpoint metadata.
    run_id: Optional[str] = None

    #: Append one entry per durable checkpoint to ``audit.jsonl`` in the store:
    #: its step, a content fingerprint, the config digest, chained by SHA-256.
    #: Off by default because it costs a hash of every checkpoint written — a
    #: file read for ``torch_save``, a ``describe()`` for Moonclip — and most
    #: runs are not asked to prove anything. See ``ravex._audit`` (GPU-93).
    audit_log: bool = False

    #: Write what ``ravex.log_metrics`` is given, and the metrics Ravex takes
    #: on its own, to ``metrics/`` in the store. See ``ravex._metrics``
    #: (GPU-147). On by default: a metric nobody logged costs nothing, and the
    #: automatic ones are one small record every ``metrics_every`` steps.
    metrics: bool = True

    #: Steps between two records of the automatic step metrics: the learning
    #: rate of every param group and the average time per step.
    metrics_every: int = 10

    #: Seconds between two samples of the machine: GPU utilisation and memory,
    #: CPU and RAM. Zero turns them off.
    system_metrics_every: float = 30.0

    #: Seconds between two metric chunks. A chunk is a file written once and
    #: never touched again, which is what lets it go up to a bucket; so this is
    #: also how late a dashboard reading the bucket sees a value, and the most
    #: a crash can lose. A week at 15 s is about 40,000 objects per process.
    metrics_chunk_every: float = 15.0

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

    def apply_storage(self, value: Any, *, strict: bool = False) -> None:
        """Take a storage section from a mapping, or from a ``StorageConfig``.

        One function for both ways in — the YAML file and the decorator's
        keyword overrides — because they had drifted into being two, and the
        second one was ``setattr(config, "storage", {...})``: the dict simply
        replaced the dataclass, and the failure arrived several frames later as
        ``AttributeError: 'dict' object has no attribute 'path'`` from inside
        ``_normalize``, naming neither ``storage`` nor the decorator. A mapping
        is the obvious thing to reach for, because that is exactly how
        ``ravex.yaml`` spells this section.

        ``strict`` is the difference between the two callers, and it is the
        distinction the rest of this module already makes. A config *file* must
        never stop a training run, so a bad section there is recorded in
        :attr:`problems` and the defaults stand. A decorator argument is
        someone typing at the call site — the most specific thing that could
        have said so — and it raises, exactly as an unknown top-level option
        already does.

        Unknown keys are refused rather than dropped, and the field list is the
        dataclass's own rather than ``hasattr``: ``is_remote`` is a property
        with no setter and ``resolve_credentials`` is a method, so ``hasattr``
        accepts both and then assigning to the first raises from a place that
        has nothing to do with configuration.
        """
        if isinstance(value, StorageConfig):
            self.storage = value
            return

        def refuse(message: str) -> None:
            if strict:
                raise TypeError(message)
            self.problems.append(message)

        if not isinstance(value, Mapping):
            refuse(
                "storage=%r is not a storage section; expected a mapping such "
                "as {'path': './checkpoints'} or a StorageConfig" % (value,)
            )
            return

        known = {f.name for f in fields(StorageConfig)}
        for key, entry in value.items():
            if key in known:
                setattr(self.storage, key, entry)
            else:
                refuse(
                    "unknown storage option %r; expected one of %s"
                    % (key, ", ".join(sorted(known)))
                )

    def _apply_mapping(self, data: Dict[str, Any]) -> None:
        if not data:
            return

        storage = data.pop("storage", None)
        if storage is not None:
            self.apply_storage(storage)

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
        if (value := get("SAVE_DTYPE")) is not None:
            self.save_dtype = _save_dtype_from_env(value)
        if (value := get("REPLICATE_EVERY")) is not None:
            self.replicate_every = _as_int(value, self.replicate_every)
        if (value := get("REPLICATION_TRANSPORT")) is not None:
            self.replication_transport = value
        if (value := get("AGREEMENT_TRANSPORT")) is not None:
            self.agreement_transport = value
        if (value := get("EMERGENCY_TRANSPORT")) is not None:
            self.emergency_transport = value
        if (value := get("KEEP_BASE_IN_MEMORY")) is not None:
            self.keep_base_in_memory = _as_bool(value, self.keep_base_in_memory)
        if (value := get("ASYNC_SAVE")) is not None:
            self.async_save = _as_bool(value, self.async_save)
        if (value := get("SHARDED_CHECKPOINTS")) is not None:
            self.sharded_checkpoints = value
        if (value := get("RESHARD_ON_RESUME")) is not None:
            self.reshard_on_resume = _as_bool(value, self.reshard_on_resume)
        if (value := get("CONVERT_FOREIGN")) is not None:
            self.convert_foreign = _as_bool(value, self.convert_foreign)
        if (value := get("TRACK_DATALOADERS")) is not None:
            self.track_dataloaders = _as_bool(value, self.track_dataloaders)
        if (value := get("TRACK_RNG")) is not None:
            self.track_rng = _as_bool(value, self.track_rng)
        if (value := get("RENDEZVOUS")) is not None:
            self.outer_rendezvous = value
        if (value := get("OUTER_JOB")) is not None:
            self.outer_job = value
        if (value := get("OUTER_MIN_NODES")) is not None:
            self.outer_min_nodes = _as_int(value, self.outer_min_nodes)
        if (value := get("HANDLE_SIGTERM")) is not None:
            self.handle_sigterm = _as_bool(value, self.handle_sigterm)
        if (value := get("EMERGENCY_COORDINATION")) is not None:
            self.emergency_coordination = _as_bool(value, self.emergency_coordination)
        if (value := get("EMERGENCY_CHECK_EVERY")) is not None:
            self.emergency_check_every = _as_int(value, self.emergency_check_every)
        if (value := get("EMERGENCY_TIMEOUT")) is not None:
            self.emergency_timeout = _as_int(value, self.emergency_timeout)
        if (value := get("OUTER_LOOP")) is not None:
            self.outer_loop = _as_bool(value, self.outer_loop)
        if (value := get("OUTER_INNER_STEPS")) is not None:
            self.outer_inner_steps = _as_int(value, self.outer_inner_steps)
        if (value := get("OUTER_ROUND_SECONDS")) is not None:
            self.outer_round_seconds = _as_float(value, self.outer_round_seconds)
        if (value := get("OUTER_LR")) is not None:
            self.outer_lr = _as_float(value, self.outer_lr)
        if (value := get("OUTER_MOMENTUM")) is not None:
            self.outer_momentum = _as_float(value, self.outer_momentum)
        if (value := get("OUTER_COMBINE")) is not None:
            self.outer_combine = value
        if (value := get("OUTER_DEADLINE")) is not None:
            self.outer_deadline = _as_int(value, self.outer_deadline)
        if (value := get("OUTER_SAVE_DTYPE")) is not None:
            self.outer_save_dtype = value or None
        if (value := get("OUTER_ROOT")) is not None:
            self.outer_root = value or None
        if (value := get("LOG_FILE")) is not None:
            self.log_file = value or None
        if (value := get("LOG_LEVEL")) is not None:
            self.log_level = value
        if (value := get("FALLBACK_ON_ERROR")) is not None:
            self.fallback_on_error = _as_bool(value, self.fallback_on_error)
        if (value := get("RUN_ID")) is not None:
            self.run_id = value
        if (value := get("AUDIT_LOG")) is not None:
            self.audit_log = _as_bool(value, self.audit_log)
        if (value := get("METRICS")) is not None:
            self.metrics = _as_bool(value, self.metrics)
        if (value := get("METRICS_EVERY")) is not None:
            self.metrics_every = _as_int(value, self.metrics_every)
        if (value := get("SYSTEM_METRICS_EVERY")) is not None:
            self.system_metrics_every = _as_float(value, self.system_metrics_every)
        if (value := get("METRICS_CHUNK_EVERY")) is not None:
            self.metrics_chunk_every = _as_float(value, self.metrics_chunk_every)

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
        None``, which crashed the runtime on construction. That crash landed
        somewhere nobody would look: the config is read before logging is
        configured, so there was no log file to say so, on a run that had asked
        for checkpoints and got none. (Under the autoloader, removed in 0.1.0,
        it was worse still - the exception was swallowed by design and the run
        carried on as though Ravex had never been asked for.)

        Bad values are replaced with defaults and recorded in ``problems``,
        which the runtime logs once it has somewhere to log to.
        """
        defaults = RavexConfig()
        numeric = (
            "checkpoint_every",
            "keep_last",
            "compression_level",
            "replicate_every",
            "emergency_check_every",
            "emergency_timeout",
            "outer_inner_steps",
            "outer_deadline",
            "metrics_every",
        )
        boolean = (
            "enabled",
            "checkpoint_on_exit",
            "resume",
            "delta",
            "keep_base_in_memory",
            "async_save",
            "reshard_on_resume",
            "convert_foreign",
            "track_dataloaders",
            "track_rng",
            "handle_sigterm",
            "emergency_coordination",
            "outer_loop",
            "fallback_on_error",
            "framework_auto_detect",
            "audit_log",
            "metrics",
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

        for name in (
            "backend",
            "compression",
            "log_level",
            "sharded_checkpoints",
            "replication_transport",
            "agreement_transport",
            "emergency_transport",
        ):
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

        if self.replication_transport not in ("auto", "sockets", "collectives"):
            # Unlike `sharded_checkpoints`, an unrecognised value here costs
            # nothing to guess wrong about: both roads move the same bytes and
            # write the same replica, so the fallback is the one that decides
            # for itself rather than the more conservative of the two.
            self.problems.append(
                f"replication_transport={self.replication_transport!r} is not "
                "'auto', 'sockets' or 'collectives'; using 'auto'"
            )
            self.replication_transport = "auto"

        if self.agreement_transport not in ("auto", "store", "collectives"):
            self.problems.append(
                f"agreement_transport={self.agreement_transport!r} is not "
                "'auto', 'store' or 'collectives'; using 'auto'"
            )
            self.agreement_transport = "auto"

        if self.emergency_transport not in ("auto", "store", "collectives"):
            self.problems.append(
                f"emergency_transport={self.emergency_transport!r} is not "
                "'auto', 'store' or 'collectives'; using 'auto'"
            )
            self.emergency_transport = "auto"

        if self.checkpoint_every < 1:
            self.checkpoint_every = 1
        if self.keep_last < 1:
            self.keep_last = 1
        if self.replicate_every < 0:
            self.replicate_every = 0
        if self.emergency_check_every < 1:
            self.emergency_check_every = 1
        if self.emergency_timeout < 1:
            self.emergency_timeout = 1
        if self.outer_inner_steps < 1:
            self.outer_inner_steps = 1
        if self.outer_round_seconds < 0:
            self.outer_round_seconds = 0.0
        if self.outer_deadline < 1:
            self.outer_deadline = 1
        if self.metrics_every < 1:
            self.metrics_every = 1
        system_every = _as_float(self.system_metrics_every, -1.0)
        if system_every < 0:
            self.problems.append(
                f"system_metrics_every={self.system_metrics_every!r} is not a "
                "number of seconds; using 30"
            )
            system_every = 30.0
        self.system_metrics_every = system_every
        chunk_every = _as_float(self.metrics_chunk_every, -1.0)
        if chunk_every < 0:
            self.problems.append(
                f"metrics_chunk_every={self.metrics_chunk_every!r} is not a "
                "number of seconds; using 15"
            )
            chunk_every = 15.0
        self.metrics_chunk_every = chunk_every
        if str(self.outer_combine) not in _OUTER_COMBINE_MODES:
            self.problems.append(
                f"outer_combine={self.outer_combine!r} is not one of "
                f"{', '.join(_OUTER_COMBINE_MODES)}; using 'mean'"
            )
            self.outer_combine = "mean"
        if self.outer_save_dtype is not None:
            # Same check `_normalize_save_dtype` makes on the checkpoint one,
            # and for the same reason: Moonclip refuses a bad target when the
            # manager is built, which for a round store is inside the `except`
            # that gives up on the outer loop and carries on training alone. A
            # typo would cost the whole run's exchange and say one line about
            # it. It matters more here than it did there now that the default
            # is a dtype rather than nothing: `none` has to keep meaning off,
            # however it arrives.
            name = str(self.outer_save_dtype).strip().lower()
            if name in ("", "none", "off"):
                # Every spelling of "off", and it needs all of them now that
                # the default is a dtype: unset no longer means uncast, so a
                # user turning it back off must not have to guess which word
                # this option takes. `compression` accepts the same three.
                self.outer_save_dtype = None
            elif name not in _SAVE_DTYPES:
                self.problems.append(
                    f"outer_save_dtype={self.outer_save_dtype!r} is not a dtype "
                    f"Moonclip can store; sending the delta uncast. Use one of: "
                    f"{', '.join(sorted(_SAVE_DTYPES))}"
                )
                self.outer_save_dtype = None
            else:
                self.outer_save_dtype = name
        if self.outer_rendezvous is not None:
            from ravex._dist.rendezvous import parse_address

            text = str(self.outer_rendezvous).strip()
            try:
                parse_address(text)
                self.outer_rendezvous = text
            except ValueError as exc:
                # Recorded rather than raised, like every other option read
                # from a file. The outer loop then looks for torch's store, does
                # not find one, and says so — so a typo here is two log lines,
                # not a node quietly training alone.
                if text:
                    self.problems.append(f"outer_rendezvous: {exc}; not using it")
                self.outer_rendezvous = None
        if not isinstance(self.outer_job, str):
            self.outer_job = str(self.outer_job)
        from ravex._dist.rendezvous import valid_job

        if not valid_job(self.outer_job):
            self.problems.append(
                f"outer_job={self.outer_job!r} is not a job name (letters, "
                "digits, '.', '_' and '-'); using 'default'"
            )
            self.outer_job = "default"
        if self.outer_min_nodes < 1:
            self.outer_min_nodes = 1
        if str(self.compression).strip().lower() in ("none", "off", ""):
            self.compression_level = 0

        # A remote store without a bucket is a misconfiguration; degrade to
        # local rather than failing the run.
        if self.storage.is_remote and not self.storage.bucket:
            self.storage.type = "local"

        self._normalize_save_dtype()

        if self.run_id and not self.storage.prefix:
            self.storage.prefix = self.run_id

    def _normalize_save_dtype(self) -> None:
        """Check ``save_dtype`` at load time, and drop what cannot be honoured.

        Moonclip refuses a bad target too, but it does so when the manager is
        *constructed* — which under Ravex is mid-run, inside the ``except``
        that falls back to ``torch.save``. A typo would end up costing the
        whole run's Moonclip checkpointing and saying one line about it. Here
        it lands in ``problems`` next to every other bad value, before
        anything starts.

        A rule with an unusable dtype is dropped rather than the whole
        setting: the other rules were spelled correctly and there is no reason
        to punish them. A dropped rule means that component is stored
        unchanged, which is the safe direction.
        """
        value = self.save_dtype
        if value is None:
            return

        if isinstance(value, str):
            name = value.strip().lower()
            if name not in _SAVE_DTYPES:
                self.problems.append(
                    f"save_dtype={value!r} is not a dtype Moonclip can store; "
                    f"ignoring it. Use one of: {', '.join(sorted(_SAVE_DTYPES))}"
                )
                self.save_dtype = None
            else:
                self.save_dtype = None if name == "none" else name
            return

        if not isinstance(value, dict):
            self.problems.append(
                f"save_dtype={value!r} is neither a dtype nor a mapping of "
                "component to dtype; ignoring it"
            )
            self.save_dtype = None
            return

        kept: Dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            dtype = str(raw_value).strip().lower()

            if dtype not in _SAVE_DTYPES:
                self.problems.append(
                    f"save_dtype[{key!r}]={raw_value!r} is not a dtype "
                    f"Moonclip can store; dropping that rule. Use one of: "
                    f"{', '.join(sorted(_SAVE_DTYPES))}"
                )
                continue

            # A component name, or a glob. Anything else is neither: Ravex
            # names every tensor under `ravex/`, so a bare word that is not a
            # component can only match a tensor called exactly that, which
            # does not exist. `{"weights": "bf16"}` is the shape of the
            # mistake — right idea, wrong noun — and it would otherwise cast
            # nothing at all and never say so.
            if key.lower() in _SAVE_DTYPE_COMPONENTS:
                kept[key.lower()] = dtype
            elif "*" in key or "/" in key:
                kept[key] = dtype
            else:
                self.problems.append(
                    f"save_dtype key {key!r} is neither a component "
                    f"({', '.join(sorted(_SAVE_DTYPE_COMPONENTS))}) nor a "
                    "pattern containing '*' or '/'; dropping that rule"
                )

        self.save_dtype = kept or None

    def resolve_save_dtype(self) -> Optional[Union[str, Dict[str, str]]]:
        """``save_dtype`` in the form Moonclip takes: globs, not components.

        Component names expand **in place**, so the order the rules were
        written in survives — and it has to, because Moonclip applies the
        first rule that matches. ``{"model": "none", "*": "bf16"}`` becomes
        three rules with the two model globs still ahead of the catch-all,
        which is what keeps it meaning "everything except the weights".

        Returns ``None`` when nothing is configured, so the caller can leave
        the argument out entirely rather than pass a value that means the
        same as not passing one.
        """
        value = self.save_dtype
        if value is None or isinstance(value, str):
            return value

        expanded: Dict[str, str] = {}
        for key, dtype in value.items():
            for pattern in _SAVE_DTYPE_COMPONENTS.get(key, (key,)):
                expanded.setdefault(pattern, dtype)
        return expanded or None

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
