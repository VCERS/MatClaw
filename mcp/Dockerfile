# MatClaw MCP server — production image (streamable-http transport)
#
# Build context is this directory (mcp/), not the repo root:
#   docker build -t matclaw-mcp mcp/
#
# GPU: the image carries no CUDA toolkit of its own — the torch wheels bundle
# the CUDA runtime libraries, and libcuda comes from the host via the NVIDIA
# container toolkit. Run it with a device reservation (see the compute_layer
# docker-compose.yml in VCERS-deployment). With no GPU visible, or with
# MATCLAW_FORCE_CPU=1, get_torch_device() falls back to CPU and tools still run.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# cu126 is the lowest CUDA wheel index that ships torch 2.10.0 (cu121 and cu124
# stop earlier). It runs on the DGX's CUDA 12.2 driver through CUDA minor
# version compatibility, which needs driver >= 525.60.13 — check with nvidia-smi.
ARG TORCH_VERSION=2.10.0
ARG TORCH_CUDA=cu126
ARG REQUIREMENTS=requirements.prod.txt

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# g++/gcc are needed by a few sdist-only scientific dependencies; they are
# removed again below so they don't ship in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch first, from the CUDA index. The PyPI wheel is CPU-only, so
# installing it here (rather than letting matgl/matcalc pull it in) is what
# makes the image GPU-capable. The index mirrors torch's own dependencies.
RUN pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
        "torch==${TORCH_VERSION}+${TORCH_CUDA}"

COPY ${REQUIREMENTS} ./
RUN pip install -r ${REQUIREMENTS} \
    && apt-get purge -y --auto-remove build-essential

COPY . .

# Non-root, with a writable HOME for the model caches below. The cache
# directories must exist and be owned by `app` in the image: Docker seeds a new
# named volume from the image's directory at the mount point, so if they were
# created by the mount instead they would come out root-owned and the server
# would fail to write to them.
RUN useradd --create-home --shell /bin/bash --uid 1001 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app
USER app

# matgl and matcalc download pretrained weights on first use. Point every cache
# at one directory so a single volume mount survives container restarts —
# without it, every restart re-downloads several hundred MB of model files.
ENV HOME=/home/app \
    XDG_CACHE_HOME=/home/app/.cache \
    HF_HOME=/home/app/.cache/huggingface

# Production toolset. Override to widen it, but add the matching dependencies
# to requirements.prod.txt first — see server.py's TOOL_GROUPS for valid names.
ENV MATCLAW_ENABLED_GROUPS=matgl,matcalc

ENV MCP_PORT=8500 \
    MCP_WORKERS=1

EXPOSE 8500

# FastMCP exposes no health endpoint, so probe the listening socket instead.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os,socket; socket.create_connection(('127.0.0.1', int(os.environ['MCP_PORT'])), 5).close()" || exit 1

CMD ["sh", "-c", "exec python server.py --transport streamable-http --host 0.0.0.0 --port ${MCP_PORT} --workers ${MCP_WORKERS}"]
