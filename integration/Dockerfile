# Integration test environment.
#
# The unit suite runs in-process: it opens the runtime directly and simulates a
# crash with an exception. That cannot exercise the parts that only exist across
# a real process boundary — config discovery from the working directory, a
# resume that starts from a genuinely empty interpreter, and a crash delivered
# as SIGKILL, where no `finally` and no atexit hook runs. Nor can it run
# torchrun.
#
# So: a Linux container, ravex installed for real, and training scripts that are
# ordinary scripts with one decorator on `main`.
#
# Until GPU-108 the point of this image was narrower and larger at once: the
# `.pth` autoloader, which attached Ravex to scripts containing no reference to
# it. That is gone. What is left is still the only place several of these
# questions can be asked.
#
#   docker build -f integration/Dockerfile -t ravex-integration .
#   docker run --rm ravex-integration

FROM python:3.12-slim

# CPU wheels only: these tests are about process lifecycle, not throughput,
# and the CUDA build is 2 GB of download for nothing.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 \
    && pip install --no-cache-dir pytest numpy

# The frameworks Ravex claims to support by virtue of hooking PyTorch itself.
# That claim is an argument until something runs it, and both of these own the
# training loop and wrap the dataloader in ways plain PyTorch does not.
# accelerate is what Trainer actually runs on, and it is what wraps the
# dataloader in a subclass of its own.
RUN pip install --no-cache-dir "transformers>=4.40" "accelerate>=1.1" "lightning>=2.2"

# DeepSpeed, for GPU-90. Not to run training on it — to *produce* real ZeRO
# checkpoints and to check Ravex's reader against DeepSpeed's own
# `zero_to_fp32`, which every checkpoint carries inside it. An adapter written
# against a format read about rather than held is how you get a parser for a
# plausible file nobody ever wrote, and this is what makes the artefacts
# available without a GPU: DeepSpeed selects its CPU accelerator on its own
# when it finds no device, and ZeRO's partitioning is Python.
#
# `g++` is not optional here. The CPU accelerator JIT-builds a shared-memory
# communication extension on first use, and without a compiler the failure is
# an ImportError about a `.so` that was never built — which reads like a broken
# install rather than a missing build tool.
RUN apt-get update && apt-get install -y --no-install-recommends g++ curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir deepspeed

# Rust, since GPU-105: the reshard planner is a compiled extension and
# `pip install -e .` builds it. Debian's rustc is far too old and
# `rust-toolchain.toml` pins 1.97.1, so that exact version is installed here —
# naming a different one only makes rustup fetch a second compiler to satisfy
# the toolchain file.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.97.1 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
# The crate. `Cargo.lock` is copied too: the image is a test environment, and a
# dependency resolving differently here than in CI is a difference nobody asked
# for and nobody would look for.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
COPY ravex ./ravex
COPY tests ./tests
COPY integration ./integration

RUN pip install --no-cache-dir -e .

# No autoloader to install since GPU-108; `status` stays because a broken
# install — most likely a compiled core that did not build — is worth failing
# on here rather than inside the first test.
RUN ravex status

CMD ["pytest", "integration", "-v", "--tb=short"]
