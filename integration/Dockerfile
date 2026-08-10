# Integration test environment.
#
# The unit suite runs in-process: it calls ravex.activate() and simulates a
# crash with an exception. That cannot exercise the parts that only exist
# across a real process boundary — the .pth autoloader, config discovery from
# the working directory, and a resume that starts from a genuinely empty
# interpreter. Nor can it run torchrun.
#
# So: a Linux container, ravex installed for real, `ravex enable` writing the
# autoloader into site-packages, and training scripts that contain not one
# reference to ravex.
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

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY ravex ./ravex
COPY tests ./tests
COPY integration ./integration

RUN pip install --no-cache-dir -e .

# The real thing: one line in site-packages, executed at interpreter startup.
# From here on nothing needs to import ravex explicitly.
RUN ravex enable && ravex status

CMD ["pytest", "integration", "-v", "--tb=short"]
