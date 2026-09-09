# Base image with PyTorch and CUDA support
# Using PyTorch 2.2 with CUDA 11.8 (matches your SRT cluster environment)
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install basic system dependencies required by some scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install standard build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the project files into the container
COPY . /app/

# Install the package with all optional dependencies
RUN pip install --no-cache-dir -e ".[train]"

# Create common directories for mounting volumes
RUN mkdir -p /app/data /app/checkpoints /app/results

# Default command: display rst-infer help
# Users can override this by passing a command to `docker run`
CMD ["rst-infer", "--help"]
