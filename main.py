
# --------------------------------------------------
# 1. IMPORTS
# --------------------------------------------------
import os
import io
import uuid
import tempfile
import logging
from typing import Optional

import numpy as np
import librosa
from pydub import AudioSegment
from scipy.spatial.distance import cosine

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import whisper
from resemblyzer import VoiceEncoder, preprocess_wav
import torch

from TTS.api import TTS

# --------------------------------------------------
# 2. LOGGING SETUP
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("voice-api")

# --------------------------------------------------
# 3. FASTAPI APP INIT
# --------------------------------------------------
app = FastAPI(
    title="Voice Mimicry Research API",
    version="2.0"
)

# --------------------------------------------------
# 4. OUTPUT DIRECTORY
# --------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

# --------------------------------------------------
# 5. LOAD MODELS (ONCE AT STARTUP)
# --------------------------------------------------
logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("small")

logger.info("Loading speaker encoder...")
voice_encoder = VoiceEncoder()

logger.info("Loading Coqui XTTS model (this may take time on first run)...")
try:
    tts_model = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        gpu=False  # ✅ safer on Windows unless CUDA is confirmed
    )
except Exception as e:
    logger.error(f"Failed to load TTS model: {e}")
    raise RuntimeError("Failed to load TTS model") from e

# --------------------------------------------------
# 6. AUDIO PREPROCESSING
# --------------------------------------------------
def convert_to_wav(audio_bytes: bytes) -> str:
    """Convert uploaded audio to mono 16kHz WAV"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.close()

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(temp_file.name, format="wav")

    return temp_file.name

# --------------------------------------------------
# 7. PROSODY EXTRACTION
# --------------------------------------------------
def extract_prosody(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path, sr=16000)

    pitches, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    pitches = pitches[~np.isnan(pitches)]

    return {
        "mean_pitch": float(np.mean(pitches)) if len(pitches) else 0.0,
        "energy": float(np.mean(librosa.feature.rms(y=y))),
        "duration_sec": float(len(y) / sr)
    }

# --------------------------------------------------
# 8. SPEAKER EMBEDDING
# --------------------------------------------------
def extract_embedding(wav_path: str):
    wav = preprocess_wav(wav_path)
    return voice_encoder.embed_utterance(wav)

# --------------------------------------------------
# 9. SIMILARITY SCORE
# --------------------------------------------------
def similarity_score(original: str, generated: str) -> float:
    e1 = extract_embedding(original)
    e2 = extract_embedding(generated)
    return float(np.clip((1 - cosine(e1, e2)) * 100, 0, 100))

# --------------------------------------------------
# 10. MAIN API ENDPOINT
# --------------------------------------------------
@app.post("/process-audio")
async def process_audio(
    audio: UploadFile = File(...),
    text_override: Optional[str] = Form(None)
):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Audio received")

    try:
        audio_bytes = await audio.read()
        input_wav = convert_to_wav(audio_bytes)

        # -----------------------------
        # Speech Recognition (Whisper)
        # -----------------------------
        logger.info(f"[{request_id}] Running Whisper ASR")
        result = whisper_model.transcribe(input_wav)

        transcript = result["text"].strip()
        language = result["language"]

        if not transcript:
            return JSONResponse({"error": "No speech detected"}, status_code=400)

        # -----------------------------
        # Prosody Analysis
        # -----------------------------
        prosody = extract_prosody(input_wav)

        # -----------------------------
        # Text to Generate
        # -----------------------------
        text_to_speak = text_override if text_override else transcript

        # -----------------------------
        # Voice Generation (XTTS)
        # -----------------------------
        logger.info(f"[{request_id}] Generating synthetic voice")

        output_filename = f"generated_{request_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        tts_model.tts_to_file(
            text=text_to_speak,
            speaker_wav=input_wav,
            language=language,
            file_path=output_path
        )

        # -----------------------------
        # Similarity Score
        # -----------------------------
        score = similarity_score(input_wav, output_path)
        audio_url = f"http://127.0.0.1:8000/audio/{output_filename}"

        logger.info(f"[{request_id}] Done | Similarity: {score:.2f}%")

        return {
            "request_id": request_id,
            "language": language,
            "transcript": transcript,
            "prosody": prosody,
            "similarity_score": score,
            "passed_75_percent": score >= 75,
            "audio_url": audio_url
        }

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# --------------------------------------------------
# 11. RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
