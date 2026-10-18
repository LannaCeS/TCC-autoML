FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    swig \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# auto-sklearn 0.15's ConfigSpace extension is not compatible with NumPy 2.
# Pin the whole legacy stack so future rebuilds resolve the same ABI-compatible set.
RUN uv pip install --system --no-cache \
    jupyterlab \
    ipykernel \
    pandas \
    numpy==1.26.4 \
    scikit-learn==0.24.2 \
    ConfigSpace==0.4.21 \
    auto-sklearn==0.15.0

WORKDIR /workspace
