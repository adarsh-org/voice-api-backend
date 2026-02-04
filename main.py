"""
Voice API Backend
FastAPI server for voice synthesis and cloning.
Integrates with VoiceClaw GPU Worker for XTTS inference.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import os
import hashlib
import time
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice API",
    description="AI Voice Synthesis & Cloning API",
    version="0.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Configuration ============

# GPU Worker URL - VoiceClaw XTTS Worker
XTTS_WORKER_URL = os.getenv("XTTS_WORKER_URL", "http://34.172.222.123:8080")
XTTS_TIMEOUT = int(os.getenv("XTTS_TIMEOUT", "60"))  # seconds

# ============ Models ============

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: str = Field(..., description="Voice ID to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    format: str = Field(default="wav", pattern="^(mp3|wav|ogg)$", description="Output format")
    language: str = Field(default="en", description="Language code for synthesis")

class GenerateResponse(BaseModel):
    id: str
    status: str
    audio_url: Optional[str] = None
    duration_ms: Optional[int] = None
    characters: int
    created_at: str

class VoiceCloneRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    # audio_file would be uploaded separately via multipart

class Voice(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    language: str
    gender: str
    preview_url: Optional[str] = None
    is_cloned: bool
    created_at: str

class UsageResponse(BaseModel):
    period_start: str
    period_end: str
    characters_used: int
    characters_limit: Optional[int]
    requests_count: int
    voices_count: int

class APIKey(BaseModel):
    key: str
    name: str
    created_at: str
    last_used: Optional[str]

# ============ Mock Data ============

MOCK_USAGE = {
    "characters_used": 12450,
    "characters_limit": 100000,
    "requests_count": 156,
    "voices_count": 3
}

# ============ GPU Worker Client ============

async def get_gpu_worker_health() -> dict:
    """Check GPU worker health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{XTTS_WORKER_URL}/health")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"GPU worker health check failed: {e}")
    return {"status": "unavailable", "gpu_available": False}

async def get_gpu_worker_voices() -> List[dict]:
    """Fetch voices from GPU worker."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{XTTS_WORKER_URL}/v1/voices")
            if response.status_code == 200:
                data = response.json()
                return data.get("voices", [])
    except Exception as e:
        logger.error(f"Failed to fetch GPU worker voices: {e}")
    return []

async def synthesize_with_gpu_worker(text: str, voice_id: str, language: str = "en", speed: float = 1.0) -> Optional[bytes]:
    """
    Call GPU worker to synthesize speech.
    Returns WAV audio bytes or None on failure.
    """
    try:
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "speed": speed
        }
        
        logger.info(f"Calling GPU worker: {XTTS_WORKER_URL}/v1/synthesize/stream")
        logger.info(f"Payload: {payload}")
        
        async with httpx.AsyncClient(timeout=XTTS_TIMEOUT) as client:
            response = await client.post(
                f"{XTTS_WORKER_URL}/v1/synthesize/stream",
                json=payload
            )
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"GPU worker returned {response.status_code}: {response.text}")
                return None
                
    except httpx.TimeoutException:
        logger.error(f"GPU worker request timed out after {XTTS_TIMEOUT}s")
        return None
    except Exception as e:
        logger.error(f"GPU worker synthesis failed: {e}")
        return None

# ============ Auth ============

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verify API key from header."""
    # In production: lookup key in database, check rate limits, etc.
    if not x_api_key or len(x_api_key) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Mock: accept any key starting with "va_"
    if not x_api_key.startswith("va_"):
        raise HTTPException(status_code=401, detail="Invalid API key format")
    
    return x_api_key

# ============ Routes ============

@app.get("/")
async def root():
    return {
        "name": "Voice API",
        "version": "0.2.0",
        "status": "operational",
        "docs": "/docs",
        "gpu_worker": XTTS_WORKER_URL
    }

@app.get("/health")
async def health():
    """Health check including GPU worker status."""
    gpu_health = await get_gpu_worker_health()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.2.0",
        "gpu_worker": {
            "url": XTTS_WORKER_URL,
            "status": gpu_health.get("status", "unknown"),
            "gpu_available": gpu_health.get("gpu_available", False),
            "gpu_name": gpu_health.get("gpu_name"),
            "tts_model_loaded": gpu_health.get("tts_model_loaded", False)
        }
    }

# ---- Voice Generation ----

@app.post("/v1/voice/generate")
async def generate_voice(
    request: GenerateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate speech from text using GPU worker.
    Returns WAV audio directly.
    """
    request_id = f"gen_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"[{request_id}] Generate request: voice={request.voice_id}, text_len={len(request.text)}")
    
    # Call GPU worker
    audio_data = await synthesize_with_gpu_worker(
        text=request.text,
        voice_id=request.voice_id,
        language=request.language,
        speed=request.speed
    )
    
    if audio_data is None:
        logger.error(f"[{request_id}] GPU worker failed, returning error")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "synthesis_failed",
                "message": "GPU worker unavailable or synthesis failed. Please try again.",
                "request_id": request_id
            }
        )
    
    logger.info(f"[{request_id}] Synthesis successful, returning {len(audio_data)} bytes")
    
    # Return audio directly as WAV
    return Response(
        content=audio_data,
        media_type="audio/wav",
        headers={
            "X-Request-ID": request_id,
            "X-Characters": str(len(request.text)),
            "Content-Disposition": f'attachment; filename="{request_id}.wav"'
        }
    )

@app.post("/v1/voice/generate/json", response_model=GenerateResponse)
async def generate_voice_json(
    request: GenerateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate speech from text - returns JSON metadata.
    For clients that need structured response (audio returned as base64 in future).
    """
    request_id = f"gen_{uuid.uuid4().hex[:12]}"
    
    # Call GPU worker
    audio_data = await synthesize_with_gpu_worker(
        text=request.text,
        voice_id=request.voice_id,
        language=request.language,
        speed=request.speed
    )
    
    if audio_data is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "synthesis_failed", 
                "message": "GPU worker unavailable or synthesis failed",
                "request_id": request_id
            }
        )
    
    # Estimate duration (WAV at 24kHz, 16-bit mono = 48000 bytes/sec)
    duration_ms = int((len(audio_data) / 48000) * 1000)
    
    return GenerateResponse(
        id=request_id,
        status="completed",
        audio_url=None,  # Could upload to storage and return URL
        duration_ms=duration_ms,
        characters=len(request.text),
        created_at=datetime.utcnow().isoformat()
    )

# ---- Voices ----

@app.get("/v1/voices")
async def list_voices(
    api_key: str = Depends(verify_api_key),
    include_cloned: bool = True
):
    """List available voices from GPU worker."""
    gpu_voices = await get_gpu_worker_voices()
    
    if not gpu_voices:
        # Fallback to built-in voices if GPU worker unavailable
        return [
            Voice(
                id="ana_florence",
                name="Ana Florence",
                description="Default XTTS voice - clear female voice",
                language="multi",
                gender="female",
                preview_url=None,
                is_cloned=False,
                created_at="2026-01-01T00:00:00Z"
            )
        ]
    
    # Transform GPU worker voices to our format
    voices = []
    for v in gpu_voices:
        voices.append(Voice(
            id=v.get("id"),
            name=v.get("name"),
            description=f"XTTS voice - {v.get('language', 'multi')}",
            language=v.get("language", "multi"),
            gender=v.get("gender", "unknown"),
            preview_url=None,
            is_cloned=False,
            created_at="2026-01-01T00:00:00Z"
        ))
    
    return voices

@app.get("/v1/voices/{voice_id}", response_model=Voice)
async def get_voice(
    voice_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get details of a specific voice."""
    gpu_voices = await get_gpu_worker_voices()
    
    for v in gpu_voices:
        if v.get("id") == voice_id:
            return Voice(
                id=v.get("id"),
                name=v.get("name"),
                description=f"XTTS voice - {v.get('language', 'multi')}",
                language=v.get("language", "multi"),
                gender=v.get("gender", "unknown"),
                preview_url=None,
                is_cloned=False,
                created_at="2026-01-01T00:00:00Z"
            )
    
    raise HTTPException(status_code=404, detail="Voice not found")

@app.post("/v1/voices/clone", response_model=Voice)
async def clone_voice(
    request: VoiceCloneRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Clone a voice from uploaded audio.
    
    In production: accepts multipart form with audio file.
    Currently returns placeholder.
    """
    voice_id = f"voice_{uuid.uuid4().hex[:8]}"
    
    return Voice(
        id=voice_id,
        name=request.name,
        description=request.description,
        language="en",
        gender="unknown",
        preview_url=None,
        is_cloned=True,
        created_at=datetime.utcnow().isoformat()
    )

@app.delete("/v1/voices/{voice_id}")
async def delete_voice(
    voice_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a cloned voice."""
    # Would check if voice is cloned and delete from storage
    return {"status": "deleted", "voice_id": voice_id}

# ---- Usage ----

@app.get("/v1/usage", response_model=UsageResponse)
async def get_usage(
    api_key: str = Depends(verify_api_key)
):
    """Get current usage statistics."""
    now = datetime.utcnow()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    return UsageResponse(
        period_start=period_start.isoformat(),
        period_end=now.isoformat(),
        characters_used=MOCK_USAGE["characters_used"],
        characters_limit=MOCK_USAGE["characters_limit"],
        requests_count=MOCK_USAGE["requests_count"],
        voices_count=MOCK_USAGE["voices_count"]
    )

# ---- API Keys (for dashboard) ----

@app.get("/v1/api-keys", response_model=List[APIKey])
async def list_api_keys(
    api_key: str = Depends(verify_api_key)
):
    """List API keys for the authenticated user."""
    return [
        APIKey(
            key="va_****" + api_key[-4:],
            name="Default",
            created_at="2026-01-15T00:00:00Z",
            last_used=datetime.utcnow().isoformat()
        )
    ]

@app.post("/v1/api-keys", response_model=APIKey)
async def create_api_key(
    name: str = "New Key",
    api_key: str = Depends(verify_api_key)
):
    """Create a new API key."""
    new_key = f"va_{uuid.uuid4().hex}"
    
    return APIKey(
        key=new_key,
        name=name,
        created_at=datetime.utcnow().isoformat(),
        last_used=None
    )

# ============ Error Handlers ============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": uuid.uuid4().hex[:12]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
