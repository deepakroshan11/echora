# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Install system dependencies
# ffmpeg: required for audio/video processing
# git: required by some pip packages that install from git sources
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (for better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper 'base' model at build time so it's baked into the image.
# This prevents slow cold-start model downloads on Hugging Face Spaces.
RUN python -c "import whisper; whisper.load_model('base'); print('Whisper base model downloaded OK')"

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p uploads jobs

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Set environment defaults (override in HF Space settings)
ENV TRANSLATE_URL=http://localhost:5000/translate
ENV UPLOADS_DIR=uploads
ENV JOBS_DIR=jobs
ENV MAX_FILE_SIZE_MB=500

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
