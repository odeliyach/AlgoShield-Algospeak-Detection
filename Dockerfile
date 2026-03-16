# AlgoShield — Containerised inference environment
# Build:  docker build -t algoshield .
# Run:    docker run --rm -v $(pwd)/data:/app/data algoshield \
#             python scripts/exp1/finetune.py \
#               --train_path data/exp1/train_2x.parquet \
#               --val_path   data/exp1/val_2x.parquet   \
#               --test_path  data/exp1/test_2x.parquet  \
#               --writing_params_path /app/models/output \
#               --eval_only

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for better layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies (CPU-only torch to keep image size manageable)
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default: run the test suite so the image is self-verifying
CMD ["pytest", "tests/", "-v"]
