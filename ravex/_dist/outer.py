"""The outer loop: many local steps, then one exchange of parameter deltas.

Everything else under ``ravex/_dist`` moves *checkpoints*. This module moves
*training*, and it exists because of one number (GPU-113): the link between two
rented boxes on different continents carries **7 MB/s**, measured on RunPod
global networking and reproduced nine days apart. An all-reduce of the
gradients of a one-billion-parameter model in bf16 moves 3 GB per node, which
on that link is **429 seconds — per step**. Synchronous data parallelism over
the internet is not slow, it is a different category of thing, and no amount of
tuning moves it.

**So communicate every H steps instead of every step.** Each node trains
locally with its own optimizer, and when the round closes the nodes exchange
the *difference between the parameters they started the round with and the ones
they hold now*. That difference is treated as a gradient — hence
:func:`pseudo_gradient` — and an outer optimizer takes one step on the average
of everybody's. The bytes on the wire per round are the same as one gradient
all-reduce; they are simply paid once per H steps. The saving is exactly H, and
H is ours to choose.

**The outer momentum is the part that is not FedAvg.** Averaging parameters
across nodes and carrying on is an old idea that loses noticeably against
synchronous training. Keeping a Nesterov momentum buffer *across rounds* and
stepping it with the averaged pseudo-gradient is what recovers most of that
gap, and it is why :class:`OuterOptimizer` exists rather than a call to
``torch.mean``.

**No transport here, on purpose.** Not one import of ``torch.distributed``:
this module takes a list of contributions and returns what to do about them, so
every rule in it is provable in one process with no rendezvous — the same split
``agreement`` keeps from ``collectives``. Who collects the contributions, with
what deadline, and what to do about a node that never sent one, is the
transport's business and lives elsewhere.

**Why fp32 outer parameters, always.** The live model may be bf16 under mixed
precision. An outer update is a small correction applied on top of a full
round's local progress, and accumulating those in bf16 — whose mantissa is
seven bits — rounds a good share of them away entirely, round after round, in a
way no test on a short run would show. The outer copy is fp32 and the write
back down to the parameter's own dtype happens once, at the end.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ravex")

#: The parameters of a model, by name, detached from the graph.
ParamMap = Dict[str, Any]

#: DiLoCo's published values, and they are a starting point rather than a
#: finding of ours: Nesterov, momentum 0.9, learning rate 0.7. Worth writing
#: down that ``lr=0.7`` is *not* small — the outer step is meant to travel most
#: of the way to the average of where the nodes went, not to nudge toward it.
DEFAULT_OUTER_LR = 0.7
DEFAULT_OUTER_MOMENTUM = 0.9

#: How contributions are combined. See :func:`combine`, which is where the
#: argument for the default lives.
COMBINE_MODES = ("mean", "normalized", "step_weighted")


def trainable(model) -> List[Tuple[str, Any]]:
    """The parameters this loop is responsible for, in a stable order.

    ``requires_grad`` and nothing else: a frozen embedding or an adapter-only
    fine-tune has parameters that no local step moves, so their delta is zero
    by construction and exchanging them would be the largest single waste on
    the slowest link in the system.
    """
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def float_buffers(model) -> List[str]:
    """Floating-point buffers, which this loop does **not** average.

    Batch-norm running statistics are the usual case. They are updated by the
    forward pass rather than by the optimizer, so they drift apart per node and
    nothing here brings them back together — each node ends up normalising with
    its own shard's statistics while sharing every weight.

    Reported rather than handled, and reported *once*, because the honest state
    of this is "known, not solved": averaging them is defensible for batch norm
    and wrong for a counter, and picking one silently is how a model converges
    slightly worse for a reason nobody can find.

    **Only the buffers a checkpoint carries.** A non-persistent buffer is not
    in the state dict, which is the module's own way of saying it is derived
    and not state — a causal attention mask is the common one, and it is
    identical on every node by construction. Warning about those means every
    transformer built the ordinary way gets a line about parameters drifting
    apart that names something that cannot drift, and a warning that cries wolf
    on the usual case is worse than no warning at all.
    """
    import torch

    carried = set(model.state_dict())
    return [
        name
        for name, buf in model.named_buffers()
        if buf is not None and torch.is_floating_point(buf) and name in carried
    ]


def snapshot(model, *, device=None) -> ParamMap:
    """A detached fp32 copy of every trainable parameter.

    ``device`` moves the copy off the accelerator. Two extra copies of the
    parameters live here — the snapshot and the outer momentum — and on a box
    chosen for how cheaply it rents, that is the difference between a model
    fitting and not. The cost is a host-device transfer per round, which
    against a round measured in minutes of network is not a cost at all.
    """
    import torch

    with torch.no_grad():
        return {
            name: p.detach().to(device=device or p.device, dtype=torch.float32).clone()
            for name, p in trainable(model)
        }


def pseudo_gradient(outer: ParamMap, model) -> ParamMap:
    """``outer - live``, per parameter. The round's whole report.

    **The sign is the point.** Subtracting this way makes the result point the
    way a gradient points: descending it moves the outer parameters *toward*
    where the local training went. An outer optimizer can then be an ordinary
    optimizer, with momentum and a learning rate that mean what they usually
    mean, instead of a bespoke averaging rule that has to be reasoned about
    from scratch every time someone touches it.

    Raises if the model no longer has a parameter the snapshot does: that is a
    model rebuilt between the start of a round and its end, and the alternative
    to raising is an outer step computed over the parameters that happen to
    match, which restores half a model and reports success.
    """
    import torch

    live = dict(trainable(model))
    missing = [name for name in outer if name not in live]
    if missing:
        raise KeyError(
            "the model no longer has %d parameter(s) the round started with "
            "(%s%s). An outer step over only the ones that still match would "
            "apply half an update and report success"
            % (
                len(missing),
                ", ".join(missing[:3]),
                ", ..." if len(missing) > 3 else "",
            )
        )

    with torch.no_grad():
        return {
            name: value - live[name].detach().to(device=value.device, dtype=value.dtype)
            for name, value in outer.items()
        }


@dataclass
class Contribution:
    """One node's report for one round.

    ``steps`` is how many optimizer steps that node actually took, which under
    a wall-clock round is not the same number on every node and is the whole
    reason this is a dataclass rather than a bare tensor dict.
    """

    delta: ParamMap
    steps: int
    node: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def combine(contributions: List[Contribution], mode: str = "mean") -> ParamMap:
    """One pseudo-gradient out of everybody's, over whoever showed up.

    The node count is ``len(contributions)`` and never a configured world size.
    A node that died during the round is simply not in the list, and the round
    closes over the rest — which is the entire fault-tolerance story of
    GPU-113, and it is a property of this line rather than of any recovery
    machinery bolted on later.

    **Why the plain mean is the default, and it is derivable rather than a
    guess.** A delta is already proportional to the work behind it: a node that
    took a thousand steps moved roughly ten times as far as one that took a
    hundred. So weighting *again* by step count counts that node twice, once in
    the size of its delta and once in its weight. Write the plain mean out with
    ``delta_i = steps_i * d_i``, where ``d_i`` is a node's average per-step
    movement::

        sum(delta_i) / N  ==  (sum(steps_i) / N) * (sum(steps_i * d_i) / sum(steps_i))

    which reads: **the step-weighted average of per-step movement, scaled to
    what a typical node's round was worth.** The work-proportional weighting
    people reach for is already inside the plain mean, exactly once.

    The other two exist to be measured against it, not because they are
    expected to win:

    ``step_weighted``
        ``sum(steps_i * delta_i) / sum(steps_i)``. Substituting the same
        expansion makes it quadratic in ``steps_i``, so a node twice as fast
        pulls four times as hard. Kept because "obviously you weight by work"
        is the first thing anyone suggests, and an arm that shows what it costs
        is worth more than an argument against it.

    ``normalized``
        every node normalised to a common step count first — one node, one
        vote, regardless of speed. Defensible when the fast nodes' data is less
        diverse than the slow nodes', and only then.

    **Measured** (``bench/outer_loop.py``, 2 nodes, one of them 5x slower, 10
    rounds of 20 inner steps), and the measurement is worth more than the
    derivation because it caught the derivation being untestable:

    ============= ============== ================
    mode          iid shards     skewed shards
    ============= ============== ================
    mean          0.0325         **2.0127**
    step_weighted **0.0283**     2.9209
    normalized    0.1274         2.8568
    ============= ============== ================

    On **iid shards** ``step_weighted`` comes out slightly *ahead* of the
    default, against the argument above. That is not the argument being wrong,
    it is the bench being blind: if every node draws from the same
    distribution, over-weighting the fast node biases the average toward data
    that says the same thing as everybody else's, so the double counting has
    nothing to bias *toward* and only shows up as moving further per round.

    Give each node its own target and the same modes separate the other way:
    ``step_weighted`` is **45% worse** than the plain mean, and worse than
    doing nothing about heterogeneity at all. Which is the reason the default
    is the plain mean, and the reason the first version of that bench should
    not have been believed.
    """
    import torch

    if not contributions:
        raise ValueError(
            "no contributions to combine. An empty round is not an outer step "
            "of zero, it is a round where nobody reported, and the caller has "
            "to decide whether to wait, shrink the group, or stop"
        )
    if mode not in COMBINE_MODES:
        raise ValueError(
            "unknown combine mode %r, expected one of %s"
            % (mode, ", ".join(COMBINE_MODES))
        )

    # **Canonical order, so every node computes the same bits** (GPU-121).
    # Floating-point addition is not associative, and each node assembles this
    # list as "mine first, then the peers that answered" — a different order on
    # every node, for the same set. The averages then differ in the last places
    # and the difference is applied to the parameters, so it does not cancel:
    # it accumulates, round after round, and the invariant the outer loop rests
    # on can only ever be checked with a tolerance that has to be guessed.
    # Sorting by node name costs nothing at these lengths and makes "every node
    # holds one model" provable with ``torch.equal``.
    contributions = sorted(contributions, key=lambda c: (c.node or "", c.steps))

    names = list(contributions[0].delta)
    for other in contributions[1:]:
        if list(other.delta) != names:
            raise KeyError(
                "the contribution from %r covers different parameters than the "
                "first one. Averaging the intersection would build one outer "
                "step out of two different models" % (other.node or "?")
            )

    # Every mode is ``scale / denominator * sum(weight_i * delta_i)``, and the
    # denominator is *not* always the sum of the weights. Getting that wrong is
    # how ``normalized`` silently became a weighted average of the deltas
    # instead of an even average of per-step movement — an answer with the
    # right shape, plausible values, and a scale off by the spread in speeds.
    if mode == "mean":
        weights = [1.0] * len(contributions)
        denominator = float(len(contributions))
        scale = 1.0
    elif mode == "step_weighted":
        weights = [float(max(c.steps, 0)) for c in contributions]
        denominator = sum(weights)
        scale = 1.0
    else:  # normalized: one node, one vote, at the scale of a round
        weights = [1.0 / c.steps if c.steps > 0 else 0.0 for c in contributions]
        voting = [c.steps for c in contributions if c.steps > 0]
        denominator = float(len(voting))
        # Back to the scale of a round rather than of a step, so the outer
        # learning rate keeps meaning the same thing across modes. Without it
        # `normalized` would be `mean` divided by the average step count, and
        # comparing the two at one learning rate would compare scales instead
        # of weightings.
        scale = (sum(voting) / len(voting)) if voting else 0.0

    if sum(weights) <= 0 or denominator <= 0:
        raise ValueError(
            "every contribution weighs zero under mode %r, with step counts %s. "
            "A round in which nobody took a step is not an update"
            % (mode, [c.steps for c in contributions])
        )

    with torch.no_grad():
        combined: ParamMap = {}
        for name in names:
            acc = None
            for weight, contribution in zip(weights, contributions):
                if weight == 0:
                    continue
                term = contribution.delta[name].to(torch.float32) * weight
                acc = term if acc is None else acc.add_(term)
            combined[name] = acc.mul_(scale / denominator)
    return combined


class OuterOptimizer:
    """Nesterov momentum over the pseudo-gradient, kept across rounds.

    Deliberately not a ``torch.optim.SGD``: that one wants ``Parameter``
    objects with ``.grad`` attached, and the outer parameters here are plain
    tensors that no autograd graph has ever touched — often on a different
    device than the model, and by design not the tensors the forward pass
    reads. Wiring them into a real optimizer means either faking gradients onto
    parameters that are not parameters, or keeping a second set of leaves alive
    that can drift from these. Forty lines of momentum is the smaller thing to
    own.

    The buffer surviving between rounds is not an implementation detail, it is
    the mechanism: it is what carries a direction several rounds agreed on
    through a round where one node dominated the average.
    """

    def __init__(
        self,
        lr: float = DEFAULT_OUTER_LR,
        momentum: float = DEFAULT_OUTER_MOMENTUM,
        nesterov: bool = True,
    ) -> None:
        if nesterov and momentum <= 0:
            raise ValueError("nesterov needs a momentum above zero, got %r" % momentum)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.buffers: ParamMap = {}
        self.rounds = 0

    def step(self, outer: ParamMap, gradient: ParamMap) -> None:
        """Move ``outer`` in place, by one outer step along ``gradient``."""
        import torch

        with torch.no_grad():
            for name, value in outer.items():
                grad = gradient[name].to(device=value.device, dtype=value.dtype)
                if self.momentum:
                    buffer = self.buffers.get(name)
                    if buffer is None:
                        buffer = grad.clone()
                    else:
                        buffer.mul_(self.momentum).add_(grad)
                    self.buffers[name] = buffer
                    if self.nesterov:
                        update = grad.add(buffer, alpha=self.momentum)
                    else:
                        update = buffer
                else:
                    update = grad
                value.add_(update, alpha=-self.lr)
        self.rounds += 1

    def state_dict(self) -> Dict[str, Any]:
        """What a checkpoint has to carry for a resume to be a continuation.

        Losing the momentum buffer at a resume is not a small loss: it throws
        away the agreement several rounds built up, and the run carries on
        looking healthy while the next few rounds re-derive it. Ravex exists to
        make a resume a continuation, so this is not optional state.
        """
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "rounds": self.rounds,
            "buffers": self.buffers,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.lr = float(state.get("lr", self.lr))
        self.momentum = float(state.get("momentum", self.momentum))
        self.nesterov = bool(state.get("nesterov", self.nesterov))
        self.rounds = int(state.get("rounds", 0))
        self.buffers = dict(state.get("buffers") or {})


class OuterLoop:
    """The round, from the training loop's point of view.

    Three calls in the training loop and nothing else::

        loop = OuterLoop(model, inner_steps=500)
        for batch in data:
            ...                       # ordinary local training
            loop.record_step()
            if loop.round_is_over():
                mine = loop.contribution()
                everyone = transport.exchange(mine)   # not this module's job
                loop.apply(everyone)

    ``inner_steps`` closes the round on a count, ``round_seconds`` on the
    clock. **The clock is the one that makes heterogeneous nodes work** — every
    node stops at the same moment having done as many steps as it could, and
    the step counts differing is then the normal case rather than a fault. Both
    may be set, and whichever comes first closes the round.
    """

    def __init__(
        self,
        model,
        *,
        inner_steps: Optional[int] = None,
        round_seconds: Optional[float] = None,
        lr: float = DEFAULT_OUTER_LR,
        momentum: float = DEFAULT_OUTER_MOMENTUM,
        nesterov: bool = True,
        combine_mode: str = "mean",
        device=None,
        node: str = "",
    ) -> None:
        if inner_steps is None and round_seconds is None:
            raise ValueError(
                "an outer loop needs a round that ends: pass inner_steps, "
                "round_seconds, or both"
            )
        if inner_steps is not None and inner_steps < 1:
            raise ValueError("inner_steps must be at least 1, got %r" % inner_steps)
        if round_seconds is not None and round_seconds <= 0:
            raise ValueError("round_seconds must be above zero, got %r" % round_seconds)
        if combine_mode not in COMBINE_MODES:
            raise ValueError(
                "unknown combine mode %r, expected one of %s"
                % (combine_mode, ", ".join(COMBINE_MODES))
            )

        self.model = model
        self.inner_steps = inner_steps
        self.round_seconds = round_seconds
        self.combine_mode = combine_mode
        self.node = node
        self.device = device
        self.optimizer = OuterOptimizer(lr=lr, momentum=momentum, nesterov=nesterov)
        self.steps_this_round = 0
        self.round_number = 0
        self._started_at = time.monotonic()

        adrift = float_buffers(model)
        if adrift:
            logger.warning(
                "%d floating-point buffer(s) (%s%s) are not exchanged between "
                "rounds. Each node keeps its own - for batch-norm statistics "
                "that means every node normalising with its own shard's while "
                "sharing every weight.",
                len(adrift),
                ", ".join(adrift[:3]),
                ", ..." if len(adrift) > 3 else "",
            )

        self.outer = snapshot(model, device=device)

    def record_step(self) -> None:
        """One local optimizer step happened."""
        self.steps_this_round += 1

    def round_is_over(self) -> bool:
        if self.inner_steps is not None and self.steps_this_round >= self.inner_steps:
            return True
        if self.round_seconds is not None:
            return (time.monotonic() - self._started_at) >= self.round_seconds
        return False

    def contribution(self) -> Contribution:
        """This node's report: what it did with the round it was given."""
        return Contribution(
            delta=pseudo_gradient(self.outer, self.model),
            steps=self.steps_this_round,
            node=self.node,
            metadata={"round": self.round_number},
        )

    def apply(self, contributions: List[Contribution]) -> Dict[str, Any]:
        """Take the outer step and start the next round.

        ``contributions`` must include this node's own. The caller assembles
        the list because the transport is what knows whether the local
        contribution reached anybody, and a node that averages itself in twice
        is a defect that surfaces only as slightly worse convergence.
        """
        gradient = combine(contributions, mode=self.combine_mode)
        self.optimizer.step(self.outer, gradient)
        self.write_back()

        steps = [c.steps for c in contributions]
        report = {
            "round": self.round_number,
            "nodes": len(contributions),
            "steps": steps,
            "slowest": min(steps),
            "fastest": max(steps),
        }
        logger.info(
            "Outer round %d closed over %d node(s), %d-%d local steps each.",
            self.round_number,
            len(contributions),
            report["slowest"],
            report["fastest"],
        )

        self.round_number += 1
        self.steps_this_round = 0
        self._started_at = time.monotonic()
        return report

    def abandon_round(self) -> None:
        """Give up on this round without applying anything, and move on.

        **The round number advances anyway, and that is the whole point.** A
        node whose exchange failed used to stay on the same number while its
        peers moved on, so from then on it was asking them for a round they had
        already retired and they were asking it for one it had not reached: two
        nodes still running, still logging closed rounds, and permanently
        unable to see each other. One failed publish was enough.

        A round is defined by the local training that filled it, not by whether
        the network agreed about it afterwards. So the counter follows the
        steps, and a round nobody could exchange is a round that happened and
        contributed nothing.
        """
        logger.warning(
            "Outer round %d abandoned after %d local step(s); the parameters "
            "this node trained stand, and it moves on to round %d so it stays "
            "in step with its peers.",
            self.round_number,
            self.steps_this_round,
            self.round_number + 1,
        )
        self.round_number += 1
        self.steps_this_round = 0
        self._started_at = time.monotonic()

    def write_back(self) -> None:
        """Put the outer parameters into the live model.

        Down to each parameter's own dtype here and nowhere else: this is the
        single point where fp32 outer state meets a possibly-bf16 model, which
        keeps the rounding in one place instead of spread through every
        arithmetic step above it.
        """
        import torch

        with torch.no_grad():
            for name, p in trainable(self.model):
                value = self.outer.get(name)
                if value is None:
                    continue
                p.data.copy_(value.to(device=p.device, dtype=p.dtype))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_number,
            "steps_this_round": self.steps_this_round,
            "combine_mode": self.combine_mode,
            "outer": self.outer,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.round_number = int(state.get("round", 0))
        self.steps_this_round = int(state.get("steps_this_round", 0))
        self.combine_mode = state.get("combine_mode", self.combine_mode)
        if state.get("outer"):
            self.outer = dict(state["outer"])
        if state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])
        self._started_at = time.monotonic()
