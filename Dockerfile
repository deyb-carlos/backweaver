
# Base image with CUDA support for GPU
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential \
    libpq-dev \
    gcc \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
# Create non-root user as per HF Spaces guidelines
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . .

# Expose the port
EXPOSE 7860

# Start the application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]