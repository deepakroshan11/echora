# ── Echora Backend Dockerfile ─────────────────────────────────────────────
# Target: Hugging Face Spaces (Docker SDK), free tier — CPU-only, ~8GB RAM
# Port 7860 is HF Spaces' expected default port.

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download Whisper "base" model at BUILD TIME so the container doesn't
# re-download on every cold start. This adds ~150 MB to the image but saves
# 30–60 seconds on every startup.
RUN python -c "import whisper; whisper.load_model('base')"

# Create upload/output directories
RUN mkdir -p uploads outputs

# Expose the port HF Spaces expects
EXPOSE 7860

# Environment defaults (override in HF Spaces Settings → Variables)
ENV TRANSLATE_URL=http://localhost:5000/translate
ENV WHISPER_MODEL=base
ENV UPLOADS_DIR=uploads
ENV OUTPUTS_DIR=outputs

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
