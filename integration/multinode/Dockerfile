# One container, one machine, one rank.
#
# Separate from `integration/Dockerfile`, which builds a much heavier image:
# the framework tests want transformers, accelerate and lightning, and none of
# those have anything to say about how a checkpoint is spread over machines.
# Six copies of that image is a lot of disk to answer a question about
# directories.
#
# moonclip comes from PyPI rather than from `../moonclip`, so what this bench
# exercises is the working tree's Ravex against the *released* engine. That is
# the honest default — it is what a user gets — and it means the image needs no
# Rust toolchain. To try an unreleased Moonclip, build a wheel with maturin and
# add it here; the difference matters for anything touching remote storage,
# which reached the released engine in 0.0.8 — before that the remote was
# push-only and there was nothing to pull back.
#
# Since GPU-105 Ravex has a compiled core, so it is built here in a stage of
# its own and only the wheel crosses over: the runtime image stays without a
# Rust toolchain, which is most of what "lighter than integration/Dockerfile"
# means. Without this stage `pip install -e .` asks maturin for a compiler that
# is not there, and the bench had not built since.
FROM python:3.12-slim AS core

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
# The version `rust-toolchain.toml` pins, or rustup fetches a second compiler.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.98.1 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir "maturin>=1.5,<2.0"

WORKDIR /build
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
COPY ravex ./ravex
RUN maturin build --release --out /wheels

FROM python:3.12-slim

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 \
    && pip install --no-cache-dir numpy

RUN pip install --no-cache-dir moonclip

COPY --from=core /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl

WORKDIR /app
COPY integration ./integration

# Ravex attaches through the decorator on the script's main(); this only
# checks the install is sound —
# same discipline as the rest of `integration/`.
RUN ravex status

CMD ["python3"]
