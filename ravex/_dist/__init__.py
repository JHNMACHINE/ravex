"""More than one GPU, and more than one machine.

The five modules here are ordered by how far the problem reaches:

:mod:`~ravex._dist.collectives`
    The process group. ``gather`` versus ``per_rank`` for a sharded model,
    ``all_ranks_agree`` for the decisions every rank has to take together, and
    the isolated short-timeout group the SIGTERM channel runs on.
:mod:`~ravex._dist.reshard`
    Resuming a per-rank checkpoint onto a different world size. Deliberately
    **pure** — no torch, no process group, no I/O, only integers — which is
    why it is the one module here with unit tests that need no launcher.
:mod:`~ravex._dist.identity`
    Who wrote a per-rank store, and as part of which run. Without it a resume
    cannot tell "never written" from "written on a machine not in this job".
:mod:`~ravex._dist.replication`
    Each rank copies its store to a peer on another machine, for the case with
    no bucket and no shared filesystem.
:mod:`~ravex._dist.elastic`
    Membership changes without a process restart.

Nothing is re-exported. ``collectives`` is the only one that pulls in
``torch.distributed`` at import, and callers reach for the submodule they
need — usually inside the function that needs it, so that a single-process run
never loads any of this.
"""
