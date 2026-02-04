# Voice API Backend

FastAPI server for voice synthesis and cloning, powered by VoiceClaw GPU Worker (XTTS).

## Features

- **Text-to-Speech** - High-quality voice synthesis using XTTS v2
- **Multiple Voices** - Support for multiple voice personas
- **Multi-language** - 16+ languages supported
- **Low Latency** - GPU-accelerated inference on Tesla T4

## Architecture

```
Client -> Voice API Backend (Railway) -> GPU Worker (GCP)
                                              |
                                        XTTS v2 Model
```

## API Endpoints

### Voice Generation

**POST /v1/voice/generate**
- Generates speech from text
- Returns WAV audio directly
- Requires `X-API-Key` header

```bash
curl -X POST https://your-api.railway.app/v1/voice/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: va_your_api_key" \
  -d '{"text": "Hello world!", "voice_id": "ana_florence", "language": "en"}' \
  -o output.wav
```

### List Voices

**GET /v1/voices**
- Returns available voices from GPU worker

### Health Check

**GET /health**
- Returns backend and GPU worker status

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `XTTS_WORKER_URL` | GPU worker URL | `http://34.172.222.123:8080` |
| `XTTS_TIMEOUT` | Request timeout (seconds) | `60` |

## Supported Languages

`en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh`, `ja`, `ko`, `hi`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --port 8000

# Test
curl http://localhost:8000/health
```

## Deployment

Deployed on Railway with auto-deploy from GitHub.

Set `XTTS_WORKER_URL` environment variable in Railway dashboard if using a different GPU worker.

## GPU Worker Endpoints

The GPU worker (VoiceClaw) exposes:
- `POST /v1/synthesize/stream` - Returns WAV audio
- `GET /v1/voices` - List available voices
- `GET /health` - Health check with GPU status
