# Multi-stage optional (kept single for simplicity)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (add more if needed for scraping/SSL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirement manifests first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY data/artists.txt ./data/artists.txt

# Default command (override in Cloud Run / docker run)
ENTRYPOINT ["python", "src/data/download_bulk_songs.py"]
# Example override:
# docker run --env GENIUS_TOKEN=... --env SPOTIFY_CLIENT_ID=... --env SPOTIFY_CLIENT_SECRET=... \
#   lyrics-downloader --artists-file data/artists.txt --target 3000 --songs-per-artist 40 \
#   --output data/processed/combined_songs_large.json
