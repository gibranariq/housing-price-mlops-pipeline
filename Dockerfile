FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --timeout=1000 \
    --retries=5 \
    -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .

# Create directories for volumes
RUN mkdir -p mlruns logs

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV MODEL_VERSION=latest

# Run the entrypoint app
CMD ["python", "app.py"]
