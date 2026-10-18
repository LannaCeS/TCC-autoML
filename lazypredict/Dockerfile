FROM python:3.11-slim
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install uv
RUN uv pip install --system jupyterlab lazypredict pandas scikit-learn xgboost lightgbm
WORKDIR /workspace