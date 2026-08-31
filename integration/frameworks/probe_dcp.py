"""What torch already gives us for reading a distributed checkpoint — GPU-90.

Megatron-core writes ``torch.distributed.checkpoint``, so does a plain FSDP
job, so "support Megatron" is mostly "support DCP". Before writing a reader,
this asks what torch can already do without one — because the cheapest
adapter is the one that turns out not to be needed.

Two questions, and they are the two that decide the design:

1. **Can the shapes be read without the tensors?** That is what a resharding
   planner needs (see GPU-101, which is the same question asked of moonclip),
   and it is the difference between planning a conversion and paying for it.
2. **Can the state be read with no model to load into?** DCP's ordinary
   ``load`` fills a state dict you already have. A converter has no model —
   it has a directory someone handed over.
"""

import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, format_utils


def main() -> None:
    print("torch", torch.__version__)
    exported = sorted(n for n in dir(format_utils) if not n.startswith("__"))
    print("format_utils:", exported)

    model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 3))
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.randn(2, 6)).sum().backward()
    optimizer.step()
    dcp.save(
        {"model": model.state_dict(), "optim": optimizer.state_dict()},
        checkpoint_id="/tmp/dcp_probe",
    )

    print("\n-- 1. shapes without tensors --")
    metadata = FileSystemReader("/tmp/dcp_probe").read_metadata()
    entries = metadata.state_dict_metadata
    print("   %d entries" % len(entries))
    for key in list(entries)[:6]:
        entry = entries[key]
        size = getattr(entry, "size", None)
        properties = getattr(entry, "properties", None)
        dtype = getattr(properties, "dtype", None)
        print("   %-34s %-16s %s" % (key, tuple(size) if size is not None else
                                     type(entry).__name__, dtype))

    print("\n-- 2. state without a model --")
    try:
        loaded = format_utils._load_state_dict_from_keys(
            checkpoint_id="/tmp/dcp_probe"
        )
        print("   returned %s with %d keys" % (type(loaded).__name__, len(loaded)))
        for key in list(loaded)[:4]:
            value = loaded[key]
            print("   %-34s %s" % (
                key,
                tuple(value.shape) if torch.is_tensor(value) else type(value).__name__,
            ))
    except Exception as exc:
        print("   FAILED: %s: %s" % (type(exc).__name__, exc))


def probe_empty_planner() -> None:
    """The version-stable way to read a checkpoint with no model to load into.

    ``_load_state_dict_from_keys`` is not in torch 2.8 and is in 2.12, so a
    reader built on it works on the machine it was written on and not on the
    container. ``_EmptyStateDictLoadPlanner`` is in both, and is what
    ``dcp_to_torch_save`` itself uses — the same route, without the file on
    the way out.
    """
    from torch.distributed.checkpoint.format_utils import (
        _EmptyStateDictLoadPlanner,
        _load_state_dict,
    )

    print("\n-- 3. the empty-planner route --")
    state: dict = {}
    try:
        _load_state_dict(
            state,
            storage_reader=FileSystemReader("/tmp/dcp_probe"),
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
        print("   %d keys" % len(state))
        for key in list(state)[:6]:
            value = state[key]
            print("   %-34s %s" % (
                key,
                tuple(value.shape) if torch.is_tensor(value) else repr(value)[:40],
            ))
    except Exception as exc:
        print("   FAILED: %s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
    probe_empty_planner()
