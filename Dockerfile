# Base Image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for NetCDF & PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Generate baseline model weights checkpoint
RUN python scripts/generate_sample_weights.py

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000
EXPOSE 8501

# Run the unified multi-service launcher
CMD ["python", "run.py"]
