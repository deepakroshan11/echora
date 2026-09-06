# 🎙️ ShortsDub

> AI-powered video dubbing for Tamil short videos — using the original speaker's cloned voice.
> **100% free · No API keys · Open source stack · $0/month**

---

## What it does

Upload a 30–60 second Tamil video (YouTube Short, Instagram Reel, etc.) and ShortsDub will:

1. **Transcribe** the Tamil speech using [OpenAI Whisper](https://github.com/openai/whisper) (runs locally)
2. **Translate** the text to your chosen language using [LibreTranslate](https://libretranslate.com) (self-hosted)
3. **Clone the speaker's voice** and synthesize the translated speech with [Coqui XTTS v2](https://github.com/coqui-ai/TTS) or Fish Speech
4. **Remux** the original video with the dubbed audio track using [FFmpeg](https://ffmpeg.org)

All models run locally or on free cloud tiers. No OpenAI API, no ElevenLabs, no Google Translate.

---

## Tech Stack

| Layer | Tool | License |
|---|---|---|
| Transcription | Whisper `base` model | MIT |
| Translation | LibreTranslate (self-hosted) | AGPL-3.0 |
| Voice Cloning + TTS | Coqui XTTS v2 (fallback) / Fish Speech (preferred) | CPML / Apache 2.0 |
| Audio/Video | FFmpeg | GPL |
| Backend | FastAPI + Uvicorn | BSD |
| Frontend | Vite + React | MIT |
| Database | SQLite (SQLAlchemy) | Public Domain |
| Backend Hosting | Hugging Face Spaces (free) | — |
| Frontend Hosting | Vercel (free) | — |

> ⚠️ **License note**: Coqui XTTS v2 model weights use the CPML license (non-commercial use only).
> For commercial use, switch to Fish Speech (Apache 2.0) or another commercially-licensed model.

---

## Supported Languages

| Language | Code |
|---|---|
| English | `en` |
| Hindi | `hi` |
| Telugu | `te` |
| Kannada | `kn` |
| Malayalam | `ml` |
| + French, German, Spanish, Chinese, Japanese, Korean, Portuguese, Arabic, Russian, Turkish |

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (for LibreTranslate)
- FFmpeg installed in PATH

### Option A: Docker Compose (recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/shortsdub.git
cd shortsdub

# Start LibreTranslate + backend together
docker-compose up --build

# In a separate terminal: start the frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

### Option B: Three terminals

**Terminal 1 — LibreTranslate:**
```bash
pip install libretranslate
libretranslate --port 5000
# Wait for "Running on http://0.0.0.0:5000" — first run downloads language models (~few minutes)
```

**Terminal 2 — Backend:**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

**Terminal 3 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Upload a Tamil video at http://localhost:5173.

### Quick API smoke tests
```bash
# Health check
curl http://localhost:8000/health

# Supported languages
curl http://localhost:8000/languages

# Test LibreTranslate
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"q":"வணக்கம்","source":"ta","target":"en"}'
# Expected: {"translatedText":"Hello"}

# Upload a video
curl -X POST http://localhost:8000/upload \
  -F "video=@test_clip.mp4" \
  -F "target_language=en"
# Returns: {"job_id": "...", "status": "queued"}

# Poll status
curl http://localhost:8000/status/YOUR_JOB_ID
```

---

## Deployment

### Architecture

```
[User Browser]
     │
     ▼
[Vercel — React Frontend]
     │  POST /upload, GET /status, GET /download
     ▼
[HF Space #1 — FastAPI Backend]
     │  POST /translate
     ▼
[HF Space #2 — LibreTranslate]
```

### Step 1: Deploy LibreTranslate (HF Space or Render)

**Option A — Hugging Face Space:**
1. Create a new Space at https://huggingface.co/new-space
2. SDK: **Docker**, Visibility: **Public**
3. In the Space's `Dockerfile`:
   ```dockerfile
   FROM libretranslate/libretranslate:latest
   EXPOSE 7860
   CMD ["libretranslate", "--host", "0.0.0.0", "--port", "7860"]
   ```
4. Note the public URL: `https://YOUR_USERNAME-libretranslate.hf.space`

**Option B — Render Free Web Service:**
- Docker image: `libretranslate/libretranslate`
- Port: `5000`
- Free tier is adequate.

### Step 2: Deploy Backend (HF Space)

1. Create another Space at https://huggingface.co/new-space
2. SDK: **Docker**, Visibility: **Public**
3. Push the `backend/` directory contents (the `Dockerfile` is already written)
4. In Space Settings → Variables:
   ```
   TRANSLATE_URL = https://YOUR_USERNAME-libretranslate.hf.space/translate
   ```
5. First build will take 5–15 minutes (downloads Whisper + XTTS models)
6. Note the Space URL: `https://YOUR_USERNAME-shortsdub-backend.hf.space`

**Push backend to HF Space:**
```bash
cd backend
git init
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/shortsdub-backend
git add .
git commit -m "Initial deploy"
git push hf main
```

### Step 3: Deploy Frontend (Vercel)

1. Push your repo to GitHub
2. Import at https://vercel.com/new
3. Set **Root Directory** to `frontend`
4. Add environment variable:
   ```
   VITE_API_URL = https://YOUR_USERNAME-shortsdub-backend.hf.space
   ```
5. Deploy — auto-redeploys on every push to `main`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRANSLATE_URL` | `http://localhost:5000/translate` | URL of your LibreTranslate instance |
| `UPLOADS_DIR` | `uploads` | Directory for uploaded videos |
| `JOBS_DIR` | `jobs` | Directory for job output files |
| `MAX_FILE_SIZE_MB` | `500` | Max upload file size in MB |
| `LIBRETRANSLATE_API_KEY` | _(empty)_ | API key if using a key-protected instance |
| `VITE_API_URL` | `http://localhost:8000` | Backend URL (set in Vercel) |

---

## Project Structure

```
shortsdub/
├── backend/
│   ├── app.py                 # FastAPI app — endpoints + pipeline orchestration
│   ├── models.py              # SQLite job tracking (SQLAlchemy)
│   ├── services/
│   │   ├── transcribe.py      # Whisper transcription
│   │   ├── translate.py       # LibreTranslate wrapper
│   │   ├── voice_clone.py     # Voice cloning (Fish Speech / XTTS v2)
│   │   └── video.py           # FFmpeg helpers (extract, trim, mux)
│   ├── requirements.txt
│   ├── Dockerfile             # For Hugging Face Spaces deploy
│   └── .dockerignore
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Three-phase state machine
│   │   ├── api.js             # All API calls centralized
│   │   ├── index.css          # Premium dark UI
│   │   └── components/
│   │       ├── UploadSection.jsx
│   │       ├── StatusTracker.jsx
│   │       └── DownloadSection.jsx
│   ├── index.html             # SEO meta tags
│   └── vercel.json            # SPA rewrite rule
├── docker-compose.yml         # Local dev: LibreTranslate + backend
└── README.md
```

---

## Known Limitations (v1)

| Limitation | Status | Notes |
|---|---|---|
| Single speaker only | By design | Multi-speaker diarization is a v2 feature |
| No lip-sync | By design | Fast cuts in shorts make this impractical anyway |
| Tamil source only | By design | Other source languages trivially added (change Whisper `source_lang`) |
| LibreTranslate translation quality | Known trade-off | Less fluent than commercial APIs; $0 alternative |
| Voice clone quality | Depends on clip | Best with clean, noise-free audio; no background music |
| CPU processing time | 60–120s for 45s clip | GPU significantly faster but not needed for free tier |
| HF Spaces cold start | 30–90s | Frontend shows a "waking up server" message |

---

## v2 Ideas (out of scope for now)

- Multi-speaker diarization (pyannote.audio)
- Batch multi-language output
- Lip-sync (Wav2Lip)
- Source language auto-detection
- Video preview player in the UI
- Background job queue (Redis/Celery) for scalability

---

## License

Code: **MIT**

Model weights:
- Whisper: MIT
- LibreTranslate: AGPL-3.0
- Coqui XTTS v2 weights: CPML (non-commercial)
- Fish Speech: Apache 2.0
- FFmpeg: GPL (tool use, not linked)
