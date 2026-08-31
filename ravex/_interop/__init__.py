"""Reading checkpoints that something other than Ravex wrote — GPU-90.

Four modules, and the split between them is the order the work happens in
rather than one module per format:

:mod:`~ravex._interop.foreign`
    *What is this directory?* Layout first, fields second, and nothing is
    opened that does not have to be.
:mod:`~ravex._interop.zero`
    Opens a DeepSpeed ZeRO store. Stages 1 and 2 share an assembly; stage 3
    does not, which was not visible from the file layouts.
:mod:`~ravex._interop.dcp`
    Opens a ``torch.distributed.checkpoint`` store — what a plain FSDP job
    writes, and what Megatron-core moved to. Torch ships the reader; what is
    here is mostly the entry point that survives a version change.
:mod:`~ravex._interop.convert`
    Puts what those two read into the shape Ravex's own resume path already
    consumes.

Nothing is re-exported here on purpose. Every caller imports the submodule it
needs, inside the function that needs it: none of these four touch torch at
import time, and a package that pulled all four in eagerly would give that up
for the convenience of a shorter import line.
"""
