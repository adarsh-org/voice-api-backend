# Voice API Backend

FastAPI backend for AI voice synthesis and cloning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000

# Or with Docker
docker build -t voice-api .
docker run -p 8000:8000 voice-api
```

## API Docs

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@host:5432/voiceapi
REDIS_URL=redis://localhost:6379
GCP_PROJECT_ID=your-project
GCS_BUCKET=voice-api-audio
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/voice/generate | Generate speech from text |
| GET | /v1/voices | List available voices |
| POST | /v1/voices/clone | Clone a voice from audio |
| GET | /v1/usage | Get usage statistics |
| GET | /health | Health check |

## Authentication

All endpoints require `X-API-Key` header:

```bash
curl -X POST https://api.voiceapi.dev/v1/voice/generate \
  -H "X-API-Key: va_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "voice_nova"}'
```

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

1. Connect GitHub repo
2. Add environment variables
3. Deploy!

---

*Note: GPU inference endpoints return placeholders until GCP workers are connected.*
