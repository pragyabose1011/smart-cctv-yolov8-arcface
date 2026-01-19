FROM debian:bookworm

# Install Python, pip, and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-opencv libgl1 libglib2.0-0 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install Python requirements safely (bypass Debian restriction)
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
