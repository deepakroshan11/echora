# Echora 🎙️

> AI-powered multilingual video dubbing and voice cloning — using the original speaker's cloned voice.
> **Total cost: $0/month.** Fully open-source stack.

---

## Tech Stack

| Layer | Tool | License |
|---|---|---|
| Transcription | Whisper `base` | MIT |
| Translation | LibreTranslate (self-hosted) | AGPL-3.0 |
| Voice Cloning + TTS | Fish Speech → Coqui XTTS v2 fallback | Apache 2.0 / CPML |
| Audio/Video | FFmpeg | GPL (tool use) |
| Backend | FastAPI + Uvicorn | BSD |
| Frontend | Vite + React | MIT |
| Database | SQLite (SQLAlchemy) | Public domain |

---

## Quick Start — Local Dev (3 terminals)

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker + Docker Compose (for LibreTranslate)
- ffmpeg on PATH (`choco install ffmpeg` / `apt install ffmpeg` / `brew install ffmpeg`)

### 1. Install backend dependencies

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

> **Note:** The `TTS` package (Coqui XTTS v2) requires ~4 GB of model files to be downloaded on first run. This is a one-time download cached in `~/.local/share/tts`.

### 2. Start LibreTranslate (Terminal 1)

```bash
# Option A — Docker (recommended)
docker-compose up libretranslate

# Option B — pip (installs globally)
pip install libretranslate
libretranslate --port 5000 --load-only en,hi,te,kn,ml,ta
```

Wait for the message: `Running on http://0.0.0.0:5000`

Test it works:
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"q":"வணக்கம்","source":"ta","target":"en"}'
# Expected: {"translatedText": "Hello"}
```

### 3. Start the backend (Terminal 2)

```bash
cd backend
uvicorn app:app --reload --port 8000
```

The first startup will load Whisper and XTTS v2 — expect 30–120 seconds.

### 4. Start the frontend (Terminal 3)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and upload a Tamil video!

---

## Full Docker Dev (single command)

```bash
docker-compose up --build
```

Then run the frontend separately:
```bash
cd frontend && npm run dev
```

---

## Deployment

### Backend → Hugging Face Spaces (Docker SDK, free tier)

1. Create a Space at https://huggingface.co/new-space
   - SDK: **Docker** · Visibility: **Public**
2. Push the `backend/` directory to the Space's git repo:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
   # Copy backend contents to repo root then push
   git subtree push --prefix backend hf main
   ```
3. Set environment variables in the Space's **Settings → Variables**:
   - `TRANSLATE_URL` = URL of your LibreTranslate deployment (see below)
   - `WHISPER_MODEL` = `base`

**Expected first build time:** 5–15 minutes (downloading models).

### LibreTranslate → Second HF Space (free tier)

1. Create another Space with SDK: **Docker**
2. Use this `Dockerfile` in that Space's root:
   ```dockerfile
   FROM libretranslate/libretranslate:latest
   ENV LT_LOAD_ONLY=en,hi,te,kn,ml,ta
   CMD ["libretranslate", "--host", "0.0.0.0", "--port", "7860"]
   ```
3. Its public URL will be `https://<username>-<space-name>.hf.space`
4. Set this as `TRANSLATE_URL` on your backend Space.

### Frontend → Vercel (free tier)

1. Import this repo at https://vercel.com/new
2. Set **Root Directory** = `frontend`
3. Set environment variable: `VITE_API_URL` = `https://<your-hf-space>.hf.space`
4. Deploy — Vercel auto-deploys on every push to `main`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRANSLATE_URL` | `http://localhost:5000/translate` | LibreTranslate endpoint |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny/base/small) |
| `UPLOADS_DIR` | `uploads` | Where uploaded videos are saved |
| `OUTPUTS_DIR` | `outputs` | Where dubbed videos are saved |
| `DB_PATH` | `shortsdub.db` | SQLite database path |

---

## Pipeline (what happens when you click "Dub Now")

```
Video upload
    ↓ ffmpeg
Audio extraction (16kHz mono WAV)
    ↓ Whisper base
Tamil transcription
    ↓ LibreTranslate
Translation to target language
    ↓ ffmpeg
Voice reference clip (first 8 sec)
    ↓ Fish Speech / XTTS v2
Dubbed audio synthesis (cloned voice)
    ↓ ffmpeg
Final video remux (original video + dubbed audio)
    ↓
Download dubbed MP4
```

---

## Known Limitations (v1, by design)

- **Single speaker only** — no diarization/multi-speaker support
- **Single target language per job** — no batch export
- **No lip-sync** — audio replacement only (shorts have fast cuts anyway)
- **LibreTranslate quality** — translations can be literal; commercial-grade fluency requires paid APIs (out of scope)
- **Voice clone quality** — depends on reference clip being clean (no background music)
- **HF Spaces cold start** — free tier sleeps after ~48h inactivity; first request after sleep takes 30–90 sec

---

## License

- Application code: MIT
- Whisper: MIT
- LibreTranslate: AGPL-3.0
- Fish Speech: Apache 2.0
- Coqui XTTS v2 (fallback): CPML (non-commercial only — flag if monetizing)
- FFmpeg: GPL (tool use only, not linked into product)
