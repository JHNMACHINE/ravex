"""Elastic membership without a process restart — GPU-94.

Three problems, kept as three separate primitives rather than one abstraction,
because each fails a different way and needs a different fix.

**Discovery** — how a process that is not yet a member of anything makes
itself known. No process group can help here: a process that has never
called ``init_process_group`` cannot be reached by any collective, isolated
or not. This is the one piece with no equivalent in GPU-92 — that channel
(``ravex._dist.collectives.emergency_group``) only ever coordinates ranks that are
already inside a live group. Discovery has to live one layer below that, on
the raw store rank 0 holds open across every membership change.

**Decision** — once every currently-live rank can see the same picture (a
new rank wants in, or one has gone quiet), they have to agree what to do
about it and when. This one *is* a straight generalisation of GPU-92: the
same isolated gloo subgroup, the same independent short timeout, the same
degrade-to-local-only contract — just carrying a small object instead of a
single bit. See :func:`topology_decision`.

**Regroup** — the mechanical part: tearing down the default process group
and rebuilding it at a new world size, with a rank that never existed before
joining the same rendezvous the original ranks re-enter. See
:func:`generation_store`, and the note on ``FSDPParamGroup``/``DeviceMesh``
below — rebuilding the *group* is the easy half of this.

**Pre-staging — why it does not use torch.distributed at all.** The obvious
design was to reuse ``ravex._dist.replication.exchange_stores`` on a small,
independent ``ProcessGroupGloo`` built by hand (without
``init_process_group``/``new_group``) between one old rank and a joining
candidate, so the old ranks' real training group is never touched while
bytes move in the background. The independent group itself works fine for a
collective — ``all_reduce`` on it returns the right answer without
disturbing the live default group at all, verified in
``tests/test_dist_elastic_prestage.py``. It does not work for
``exchange_stores``, because that function moves bytes with
``dist.isend``/``dist.recv``, and those refuse a group that was never
registered through ``new_group``: ``RuntimeError: ... is not registered,
please create group with torch.distributed.new_group API``. The private
escape hatch (``torch.distributed.distributed_c10d._register_process_group``)
does not help either — it rejects the same ``ProcessGroupGloo`` instance
with a pybind11 type-mismatch ``TypeError``, even though it is exactly the
type the function's own signature names. And the public path,
``new_group()``, requires every rank of the *current* default group to call
it collectively - which a not-yet-member candidate cannot do, since it has
no default group to be a member of. So pre-staging here is a plain,
length-prefixed TCP socket, not a torch collective of any kind - see
:func:`prestage_send` and :func:`prestage_receive`. It reuses
``ravex._dist.replication``'s manifest/skip/``StoreWriter`` machinery directly
(all of it is already plain bytes and files, with no torch dependency of its
own — only ``exchange_stores`` itself, the one function this module does
not call, wires that machinery to a process group).

**On "elastic without a restart" — what it actually buys.** ``fully_shard()``
refuses to be applied a second time to a module it has already wrapped
(torch raises ``AssertionError: Each distinct composable distributed API can
only be applied to a module once`` — not a ravex limitation, a torch one).
There is also no public API to move an already-materialised ``DTensor``
parameter from one ``DeviceMesh`` to another: :meth:`DTensor.redistribute`
changes *placements* on a fixed mesh, it does not change which ranks compose
the mesh. So a live regroup cannot mean "the wrapped module quietly changes
shape while training continues" — it means capturing every parameter's full
(unsharded) values via ``DTensor.full_tensor()``, rebuilding a fresh instance
of the module, loading those values into it, and calling ``fully_shard()``
fresh under the new mesh. That is still real: it skips the OS process exit,
the CUDA context re-init, and re-importing torch/moonclip that a genuine
restart pays for. It is not a live reshape of a running module, and nothing
downstream should be built on the assumption that it is. Verified bit-exact
across an actual world_size change in ``tests/test_dist_elastic_remesh.py``.

**Ravex performs that rebuild since GPU-110** — see :func:`regroup` at the
bottom of this module. It could not before, and the reason was written here:
ravex is only ever handed an already-constructed model, so it had no factory
to rebuild one from. ``@ravex.train_loop`` wraps the function the model is
born inside, which is exactly that factory, and this module no longer stops
at the group/rendezvous layer.

**Two rules for whoever calls ``full_tensor()`` around a regroup, found the
hard way in ``tests/test_dist_elastic_grow_end_to_end.py`` while composing all
of the above into one scenario** — neither is specific to that test, both
apply to any real caller:

1. ``full_tensor()`` redistributes a sharded ``DTensor`` and is a
   **collective**: every rank of the group must call it together, even the
   ranks that discard the result. Calling it on one rank alone does not
   raise anywhere near the mistake — it desynchronises
   ``torch.distributed``'s internal per-process group-naming counter
   between ranks, and that surfaces later, far from the real cause, as an
   unrelated-looking timeout the next time *any* rank builds a new group
   (``topology_decision`` included). Same discipline
   ``emergency_signalled`` and ``_drain_split_agreed`` already document for
   themselves, extended to a collective neither of those functions happens
   to call.
2. Never call ``full_tensor()`` — or anything else that reads a
   ``DTensor``'s mesh — on a module whose process group has already been
   torn down. After a regroup's ``destroy_process_group()``, the *old*
   module's parameters are still meshed against the world_size that no
   longer exists. Reading them then does not fail cleanly; it corrupts,
   surfacing as a baffling ``RuntimeError: narrow unexpectedly changed
   concrete size`` with no obvious connection to the real cause. Whatever
   values are needed from the old module have to be captured *before* the
   destroy, not reconstructed after it.
"""

from __future__ import annotations

import socket as _socket
import struct
from typing import Any, List, Optional

from ravex import _core

_JOIN_KEY = "gpu94/join/%d"


def generation_store(base_store, generation: int):
    """Scope ``base_store`` to one rendezvous "generation". **Load-bearing.**

    Not a convenience wrapper — without it, reusing the same raw store across
    two ``init_process_group`` calls at different world sizes fails, and
    fails *differently* depending on the platform, which is what makes it
    easy to mistake for a fluke rather than a rule. On Windows it fails
    deterministically, from an assertion inside gloo's own transport layer
    (``sizeof(impl_) == bytes.size()``, ``gloo/transport/uv/address.cc``).
    Cross-checked in a Linux container specifically to rule out a
    Windows-only artifact: it fails there too, differently and
    non-deterministically — "Connection reset by peer" from
    ``gloo/transport/tcp/pair.cc``, or an outright hang with every process
    still alive past the phase's own timeout. Both platforms trace to the
    same cause: the bare store still carries the first generation's
    handshake keys when a second ``init_process_group`` call at a different
    world_size reads it. A fresh :class:`~torch.distributed.PrefixStore` per
    generation keeps every generation's handshake keys on their own
    namespace, and that is enough — no platform-specific handling needed on
    top of it.

    See ``tests/test_dist_elastic_rendezvous.py`` for the reproduction of both
    failures and the negative test that keeps this from silently stopping
    being necessary if a future torch release changes the underlying
    behaviour.
    """
    import torch.distributed as dist

    return dist.PrefixStore("gpu94/gen%d" % generation, base_store)


def announce_join(base_store, candidate_rank: int, address: str) -> None:
    """A process that is not yet a member of anything advertises itself.

    Needs no process group — see the module docstring on why that is exactly
    the point. ``base_store`` is the same raw, persistent store
    :func:`generation_store` wraps per generation; a joiner reaches it as an
    ordinary ``TCPStore`` client pointed at rank 0's host and port, before it
    has ever called ``init_process_group``.
    """
    base_store.set(_JOIN_KEY % candidate_rank, address.encode("utf-8"))


def pending_join(base_store, candidate_rank: int) -> Optional[str]:
    """Non-blocking: has ``candidate_rank`` announced itself yet?

    ``None`` if not — deliberately not a wait. Whoever polls this (rank 0, at
    a step boundary) has training to keep doing between polls; the store's
    own ``get`` blocks until the key exists, which is the wrong shape for a
    check made once per step.
    """
    key = _JOIN_KEY % candidate_rank
    if not base_store.check([key]):
        return None
    return base_store.get(key).decode("utf-8")


def topology_decision(local_view: Any, timeout_seconds: int = 10) -> List[Any]:
    """Gather a small topology decision from every currently-live rank. **Collective.**

    Generalises GPU-92's ``emergency_signalled`` from a single bit ("does
    anyone want to checkpoint") to a small picklable object ("what does
    everyone see about the topology right now") — same isolated gloo
    subgroup (:func:`ravex._dist.collectives.emergency_group`), same independent
    short timeout, same degrade-to-local-only contract when the channel
    cannot be built. ``emergency_group`` itself needed no change to support
    this: it was already general enough. What was missing was the ability to
    gather more than a bit, which is why ``_all_gather_object`` and
    ``gather_objects`` now take an optional ``group=``.

    Posted only at a step boundary, unconditionally by every live rank, for
    the same reason ``emergency_signalled`` and ``_drain_split_agreed``
    document for themselves: a collective some ranks post and others do not
    is a hang waiting for the group's own timeout, not a diagnostic that
    quietly does nothing.

    Returns every live rank's ``local_view``, in rank order, or
    ``[local_view]`` alone when no coordination channel could be built —
    already the shape a caller of ``emergency_group`` has to handle for a
    bare bool, generalised to whatever picklable object ``local_view`` is.
    """
    from ravex._dist.collectives import emergency_group, gather_objects

    usable, group = emergency_group(timeout_seconds)
    if not usable:
        return [local_view]
    return gather_objects(local_view, group=group)


def prestage_send(sock, source: str, chunk: int = 1 << 20) -> int:
    """Push ``source`` to whatever ``prestage_receive`` holds on the other
    end of ``sock``, skipping files that already match. **Not a collective.**

    See the module docstring for why this is a plain socket rather than
    ``exchange_stores``. The manifest/skip/encode logic is exactly
    ``ravex._dist.replication``'s own - reused, not reimplemented, because none
    of it depends on a process group; only the transport underneath it does.

    Reads the peer's manifest first (length-prefixed), computes what it
    already has byte-for-byte, then streams the rest. Does **not** shut the
    connection down afterwards, on purpose: ``encode_store``'s own header
    already tells :func:`prestage_receive` exactly how many files of what
    size to expect, which is what lets ``StoreWriter.complete`` end the read
    loop without a message boundary from the transport. Leaving the
    connection open is what makes several rounds over one socket possible -
    the pre-staging use case this exists for: a coarse pass while the old
    ranks are still far from the cutover, then smaller and smaller deltas as
    ``replicate_every``-style periodic re-syncs bring the candidate closer,
    without paying a new TCP handshake for every round.
    """
    return _core.prestage_send(sock.fileno(), source, chunk)


def prestage_receive(sock, destination: str) -> bool:
    """The other end of :func:`prestage_send`. **Not a collective.**

    Sends what ``destination`` already holds before reading anything, then
    feeds whatever arrives to the same :class:`~ravex._dist.replication.StoreWriter`
    ``exchange_stores`` uses - the destination ends up in exactly the same
    shape a torch-transported round would have left it in. Stops reading on
    ``writer.complete`` alone - the embedded header already says how much to
    expect, so no separate length prefix or connection close is needed for
    the body, and the connection is left open for whatever round comes next.
    An empty ``recv`` (the peer actually closed) still ends the loop rather
    than spinning, but that is the belt, not the buckle: ``prestage_send``
    does not close its end after a round precisely so the same connection
    can carry the next one.
    """
    return _core.prestage_receive(sock.fileno(), destination)


# --------------------------------------------------------------------------
# The rebuild — GPU-110. Everything above this line stops at the group and
# rendezvous layer; this is the half the module docstring said ravex could not
# do, and the reason it could not is gone.


class RegroupError(RuntimeError):
    """A regroup that would corrupt rather than fail, refused up front.

    Its own class because both mistakes it names are *silent* in torch: one
    desynchronises a counter and surfaces as an unrelated timeout much later,
    the other returns wrong bytes and surfaces as ``narrow unexpectedly changed
    concrete size``. Neither points at the caller. This does.
    """


def capture_for_regroup(model, optimizers):
    """Everything the new group has to be handed, taken **before the teardown**.

    **Collective, and every rank must call it** — including any rank that is
    about to leave. It is the first of the two rules in this module's
    docstring, and it is not enforceable from here: a rank that skips this call
    does not fail near the mistake, it desynchronises ``torch.distributed``'s
    group-naming counter and surfaces later as a timeout somewhere unrelated.

    The second rule *is* enforced. Called after ``destroy_process_group()``,
    this raises :class:`RegroupError` instead of reading parameters that are
    still meshed against a world size that no longer exists — which does not
    fail cleanly, it corrupts.

    **It is the checkpoint path, deliberately.** ``gather_sharded_state`` is
    what ``sharded_checkpoints: gather`` already uses, so a regroup captures
    exactly what a checkpoint would, in the same shape, through the same code.
    Two ways to unshard state would mean two answers to every question about
    what survives, and the second answer would be the one nobody tested.

    It also brings the **optimizer** with it, which a hand-rolled
    ``full_tensor()`` over ``named_parameters()`` does not. Weights alone make
    a regroup a restart from a good place; the Adam moments are what make it a
    continuation, which is the whole promise Ravex is built on.

    **What comes back is full on rank 0 and empty everywhere else**, and that
    is the trap this function exists to name. ``gather_sharded_state`` is the
    *writing* half of the checkpoint path, and only rank 0 writes — measured
    here on three ranks: rank 0 got 4 model keys and 2 optimizer keys, ranks 1
    and 2 got nothing at all. The restore, ``apply_sharded_state``, needs the
    opposite, and says so in its own docstring: every rank passes the full
    state. A resume satisfies that without anyone noticing, because every rank
    reads the same bytes off the same checkpoint. A regroup has no checkpoint
    to read, so it has to :func:`share_captured` — and getting that wrong does
    not raise, it deadlocks the ranks against each other inside a scatter,
    which is how it was found.
    """
    from ravex._dist import collectives

    dist = collectives._dist()
    if dist is None or not dist.is_initialized():
        raise RegroupError(
            "capture_for_regroup() needs the group the model is still meshed "
            "against. Called with no process group, the parameters would be "
            "read against a world size that no longer exists - which does not "
            "raise, it returns wrong bytes and surfaces later as 'narrow "
            "unexpectedly changed concrete size'. Capture before the teardown."
        )
    return collectives.gather_sharded_state(model, list(optimizers))


def share_captured(captured, src: int = 0):
    """Give every rank of the **current** group the full captured state.

    Called once the new group is up, never before: a rank that is joining was
    in no group to receive anything on. That ordering is what spares a grow
    from shipping state out of band — the joiner needs no queue, no pre-staged
    store and no checkpoint, only the broadcast it is now a member of.

    ``src`` must be a rank that survived the regroup and therefore holds the
    capture. Rank 0 by convention, and checked rather than assumed: a source
    holding nothing would broadcast nothing, every rank would restore an empty
    state onto a freshly built model, and the run would carry on from random
    weights looking like one that merely converges badly.
    """
    from ravex._dist import collectives

    dist = collectives._dist()
    if dist is None or not dist.is_initialized():
        raise RegroupError("share_captured() needs the new group to be up")

    if dist.get_rank() == src:
        model_state, _ = captured
        if not model_state:
            raise RegroupError(
                "rank %d is the source of the regroup broadcast and holds no "
                "captured state. The source has to be a rank that was in the "
                "old group; a rank that has just joined has nothing to share, "
                "and broadcasting nothing would put every rank back on random "
                "weights without anything failing." % src
            )

    # Not `dist.broadcast_object_list`: it decodes with `tensor.numpy()`, and
    # an image without NumPy turned this line - the one that puts the model
    # back together after a regroup - into "Numpy is not available". See
    # `collectives._broadcast_object`, which is wire-compatible with it.
    return collectives._broadcast_object(dist, captured, src)


def rebuild_after_regroup(build, captured):
    """Build the model and optimizers again under the new mesh, and load state.

    ``build`` is the factory — a callable returning ``(model, optimizers)``,
    already wrapped however this run wraps them, exactly as the training
    function would construct them. **Ravex does not shard for you**: it does
    not know whether this run wants ``fully_shard``, DDP, or neither, and
    guessing would be a second opinion about the model's construction.

    That callable is what ``@ravex.train_loop`` finally makes available.
    ``fully_shard()`` refuses to wrap a module twice and there is no API to
    move a materialised ``DTensor`` to another ``DeviceMesh``, so a regroup has
    always had to mean *rebuild*; what was missing was something to rebuild
    from. Ravex is handed a constructed model, never a way to construct one —
    until the decorator holds the function the model is born inside.

    Must be called **after** the new group is up, and it says so rather than
    scattering state onto a mesh that is still the old one.
    """
    from ravex._dist import collectives

    dist = collectives._dist()
    if dist is None or not dist.is_initialized():
        raise RegroupError(
            "rebuild_after_regroup() needs the new process group to be up: "
            "the fresh model is sharded against whatever mesh exists when it "
            "is built, and there is nothing to scatter onto without one."
        )

    model, optimizers = build()
    optimizers = list(optimizers)
    model_state, optimizer_state = captured
    collectives.apply_sharded_state(model, optimizers, model_state, optimizer_state)
    return model, optimizers


def regroup(build, base_store, generation, rank, world_size, *,
            model=None, optimizers=(), backend="gloo", timeout_seconds=60):
    """Capture, tear the group down, rebuild it at ``world_size``, reload.

    The whole dance in the order the rules require, so that a caller cannot get
    it wrong by writing the steps out again. Returns the new
    ``(model, optimizers)``.

    A rank already in the group passes ``model`` and ``optimizers``; a rank
    joining passes neither, because it has nothing to capture from. **It needs
    nothing else either** — the state reaches it in the broadcast that happens
    once the new group is up, which is why the order here is capture, destroy,
    init, share, rebuild, and not the more obvious one where the values are
    shipped to the newcomer first.

    What this buys over a real restart is worth stating plainly, because the
    name oversells it: the OS process does not exit, the CUDA context is not
    re-initialised, and torch and moonclip are not re-imported. It is **not** a
    module changing shape while training continues, and nothing downstream
    should be built as though it were.
    """
    import datetime

    from ravex._dist import collectives

    captured = ({}, {})
    if model is not None:
        captured = capture_for_regroup(model, optimizers)

    dist = collectives._dist()
    if dist is None:
        raise RegroupError(
            "regroup() needs torch.distributed, and this interpreter has no "
            "usable one. Said here rather than three lines down, where the "
            "same absence would surface as an AttributeError on None and read "
            "like a bug in ravex instead of a missing backend."
        )
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        backend,
        store=generation_store(base_store, generation),
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    return rebuild_after_regroup(build, share_captured(captured))
