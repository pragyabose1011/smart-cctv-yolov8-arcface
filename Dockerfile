FROM python:3.11-slim

# Install minimal system dependencies (keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Create models cache directory to optimize memory usage
RUN mkdir -p ./models_cache

# Install Python requirements with optimizations
RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Memory and performance optimizations
ENV PORT=5000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    OMP_NUM_THREADS=1

# Use single worker and reduce timeouts for memory efficiency
# Bind to the port provided by the environment (Render sets $PORT)
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --worker-class sync --timeout 120 --max-requests 100 --max-requests-jitter 10"]
