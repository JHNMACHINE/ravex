"""Elastic membership without a process restart — GPU-94.

Three problems, kept as three separate primitives rather than one abstraction,
because each fails a different way and needs a different fix.

**Discovery** — how a process that is not yet a member of anything makes
itself known. No process group can help here: a process that has never
called ``init_process_group`` cannot be reached by any collective, isolated
or not. This is the one piece with no equivalent in GPU-92 — that channel
(``ravex._distributed.emergency_group``) only ever coordinates ranks that are
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
design was to reuse ``ravex._replication.exchange_stores`` on a small,
independent ``ProcessGroupGloo`` built by hand (without
``init_process_group``/``new_group``) between one old rank and a joining
candidate, so the old ranks' real training group is never touched while
bytes move in the background. The independent group itself works fine for a
collective — ``all_reduce`` on it returns the right answer without
disturbing the live default group at all, verified in
``tests/test_elastic_prestage.py``. It does not work for
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
``ravex._replication``'s manifest/skip/``StoreWriter`` machinery directly
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
across an actual world_size change in ``tests/test_elastic_remesh.py``; ravex
itself does not perform the rebuild, because ravex is only ever handed an
already-constructed model — it has no factory to rebuild one from, which is
why this module stops at the group/rendezvous layer and leaves the module
rebuild to whoever owns the model's construction.

**Two rules for whoever calls ``full_tensor()`` around a regroup, found the
hard way in ``tests/test_elastic_grow_end_to_end.py`` while composing all
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


_JOIN_KEY = "gpu94/join/%d"
_LENGTH_PREFIX = struct.Struct("!Q")


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

    See ``tests/test_elastic_rendezvous.py`` for the reproduction of both
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
    subgroup (:func:`ravex._distributed.emergency_group`), same independent
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
    from ravex._distributed import emergency_group, gather_objects

    usable, group = emergency_group(timeout_seconds)
    if not usable:
        return [local_view]
    return gather_objects(local_view, group=group)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        block = sock.recv(n - len(buf))
        if not block:
            raise ConnectionError(
                "peer closed while %d of %d bytes were still expected" % (n - len(buf), n)
            )
        buf += block
    return bytes(buf)


def prestage_send(sock, source: str, chunk: int = 1 << 20) -> None:
    """Push ``source`` to whatever ``prestage_receive`` holds on the other
    end of ``sock``, skipping files that already match. **Not a collective.**

    See the module docstring for why this is a plain socket rather than
    ``exchange_stores``. The manifest/skip/encode logic is exactly
    ``ravex._replication``'s own - reused, not reimplemented, because none
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
    from ravex._replication import _parse_manifest, encode_store, store_files

    manifest_len = _LENGTH_PREFIX.unpack(_recv_exact(sock, _LENGTH_PREFIX.size))[0]
    peer_existing = (
        dict(_parse_manifest(_recv_exact(sock, manifest_len))) if manifest_len else {}
    )

    skip_names = {
        relative
        for relative, size in store_files(source)
        if peer_existing.get(relative) == size
    }
    for block in encode_store(source, chunk=chunk, skip=skip_names):
        sock.sendall(block)


def prestage_receive(sock, destination: str) -> bool:
    """The other end of :func:`prestage_send`. **Not a collective.**

    Sends what ``destination`` already holds before reading anything, then
    feeds whatever arrives to the same :class:`~ravex._replication.StoreWriter`
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
    from ravex._replication import StoreWriter, _encode_manifest, store_files

    existing_blob = _encode_manifest(store_files(destination))
    sock.sendall(_LENGTH_PREFIX.pack(len(existing_blob)) + existing_blob)

    writer = StoreWriter(destination)
    try:
        while not writer.complete:
            block = sock.recv(1 << 16)
            if not block:
                break
            writer.feed(block)
        writer.close()
        return writer.commit()
    finally:
        writer.close()
