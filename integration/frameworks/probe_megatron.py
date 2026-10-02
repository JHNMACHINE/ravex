"""What a Megatron-core checkpoint under tensor and pipeline parallelism holds,
and what Ravex's DCP reader makes of it - GPU-145.

GPU-90 established that Megatron-core writes ``torch.distributed.checkpoint``
and left one thing untried: a job with TP or PP above 1, where every rank holds
a piece of a tensor or a slice of the layers. Whether that is a reader's
problem at all depends on how the pieces are recorded - as chunks of one global
tensor, which DCP reassembles by itself, or as something Megatron-shaped that
needs a layout of its own - and that is a question for an artefact, not for
the documentation.

The oracle is Megatron. A checkpoint saved at TP=2 or PP=2 is loaded by
Megatron's own ``dist_checkpointing.load`` into the same model built at TP=1,
PP=1 - resharding being what that format exists for - and that model's state
dict is the answer. Ravex reads the same directory with no model at all, and
every tensor it returns must equal Megatron's by name.

    torchrun --nproc-per-node=2 probe_megatron.py save --tp 2 --pp 1 --out /tmp/m
    torchrun --nproc-per-node=1 probe_megatron.py check --out /tmp/m

On CPU with gloo and on GPU with NCCL alike: the layout of a checkpoint does
not depend on the device it was made on, and a CPU run is the cheap place to
learn it.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

LAYERS = 4
HIDDEN = 64
HEADS = 4
VOCAB = 128
SEQUENCE = 32


def build(tp: int, pp: int):
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp
    )
    # The same seed on every rank: with CPU initialization Megatron draws the
    # full tensor and keeps this rank's piece, so the weights do not depend on
    # the parallelism they were drawn under.
    torch.manual_seed(0)
    config = TransformerConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        # Said again here and not only to parallel_state: a stage numbers its
        # layers from the config's sizes, and without them both stages of a
        # PP=2 job save themselves as layers 0 and 1.
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB,
        max_sequence_length=SEQUENCE,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    if torch.cuda.is_available():
        model.cuda()
    return model


def save(args) -> int:
    from megatron.core import dist_checkpointing

    model = build(args.tp, args.pp)
    rank = dist.get_rank()
    os.makedirs(args.out, exist_ok=True)
    dist_checkpointing.save(model.sharded_state_dict(prefix=""), args.out)
    dist.barrier()
    if rank == 0:
        print("saved at TP=%d PP=%d:" % (args.tp, args.pp))
        for name in sorted(os.listdir(args.out)):
            print("   %-40s %d" % (name, os.path.getsize(os.path.join(args.out, name))))
    return 0


def tensors_of(value, prefix=""):
    """Every tensor in a nested state, by dotted name."""
    found = {}
    if isinstance(value, torch.Tensor):
        found[prefix] = value
    elif isinstance(value, dict):
        for key, inner in value.items():
            found.update(tensors_of(inner, "%s.%s" % (prefix, key) if prefix else str(key)))
    return found


def check(args) -> int:
    from megatron.core import dist_checkpointing

    from ravex._interop.dcp import describe, read
    from ravex._interop.foreign import identify, summary

    print("identified:", summary(identify(args.out)))

    described = describe(args.out)
    print("\nthe checkpoint's own metadata, %d entries:" % len(described))
    for key in sorted(described)[:60]:
        entry = described[key]
        print("   %-70s %s %s" % (key, tuple(entry["shape"]) if "shape" in entry else "-", entry.get("dtype", entry)))

    # Megatron's answer: the same model at TP=1, PP=1, loaded from it. Not
    # compared through ``model.state_dict()``: Megatron does not save under
    # those names. It stacks a block's layers into one tensor with the layer
    # as a leading axis (``decoder.layers.mlp.linear_fc1.weight``, [L, ...]),
    # and renames some on the way. What each local tensor is in the checkpoint
    # - its key and where it sits in the global tensor - is what its
    # ShardedTensor says, so that is what the comparison follows.
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    model = build(1, 1)
    sharded = model.sharded_state_dict(prefix="")
    pieces = {k: v for k, v in sharded.items() if isinstance(v, ShardedTensor)}
    loaded = dist_checkpointing.load(sharded, args.out)

    try:
        mine = tensors_of(read(args.out))
    except Exception as exc:
        print("\n** ravex could not read it: %s: %s **" % (type(exc).__name__, exc))
        return 1

    equal, differ, absent, stacked = [], [], [], set()
    for local, piece in sorted(pieces.items()):
        theirs = loaded[local]
        theirs = theirs.detach().cpu() if isinstance(theirs, torch.Tensor) else theirs
        have = mine.get(piece.key)
        if have is None:
            absent.append("%s (as %s)" % (local, piece.key))
            continue
        # The leading axes Megatron prepended are an index; the rest a slice.
        lead = piece.prepend_axis_num
        index = tuple(piece.global_offset[:lead]) + tuple(
            slice(offset, offset + size)
            for offset, size in zip(piece.global_offset[lead:], piece.local_shape)
        )
        if lead:
            stacked.add(piece.key)
        part = have[index].cpu()
        if tuple(part.shape) != tuple(theirs.shape) or not torch.equal(part, theirs):
            differ.append("%s (as %s%s): %s vs Megatron's %s" % (
                local, piece.key, list(piece.global_offset[:lead]), tuple(part.shape), tuple(theirs.shape)))
        else:
            equal.append(local)
    used = {piece.key for piece in pieces.values()}
    extra = sorted(k for k in mine if k not in used)

    print("\nagainst Megatron's own load at TP=1 PP=1, piece by piece:")
    print("   equal:           %d of %d" % (len(equal), len(pieces)))
    print("   stacked keys:    %d (a block's layers along a leading axis)" % len(stacked))
    print("   differ:          %d" % len(differ))
    for line in differ[:15]:
        print("      " + line)
    print("   not in ravex's:  %d" % len(absent))
    for line in absent[:15]:
        print("      " + line)
    print("   only in ravex's: %d" % len(extra))
    for line in extra[:15]:
        print("      %s %s" % (line, tuple(mine[line].shape) if hasattr(mine[line], "shape") else ""))
    ok = equal and not differ and not absent
    print("\n%s" % ("ok: every piece Megatron loads, ravex reads the same" if ok else "NOT the same: see above"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["save", "check"])
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        # Megatron's writer waits for the device before staging the tensors,
        # and broadcasts its success as a tensor on "the current device", with
        # nothing in between to say there might not be one. On CPU there is
        # nothing to wait for, and the current device is the CPU.
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.current_device = lambda: "cpu"
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    try:
        return save(args) if args.phase == "save" else check(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
