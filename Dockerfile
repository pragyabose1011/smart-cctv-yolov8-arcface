FROM python:3.11-slim

# Install minimal system dependencies (keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install Python requirements
RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Ensure PORT is set before starting gunicorn
ENV PORT=5000
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1"]

