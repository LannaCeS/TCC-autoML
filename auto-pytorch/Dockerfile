FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    swig \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch: avoids CUDA/NVIDIA downloads, which are not useful on macOS.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 \
    torchvision==0.19.1

RUN pip install --no-cache-dir \
    jupyterlab \
    pandas \
    scikit-learn==0.24.2 \
    autoPyTorch==0.2.1

WORKDIR /workspace