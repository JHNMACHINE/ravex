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
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 \
    && pip install --no-cache-dir numpy

RUN pip install --no-cache-dir moonclip

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY ravex ./ravex
COPY integration ./integration

RUN pip install --no-cache-dir -e .

# Ravex attaches through the decorator on the script's main(); this only
# checks the install is sound —
# same discipline as the rest of `integration/`.
RUN ravex status

CMD ["python3"]
