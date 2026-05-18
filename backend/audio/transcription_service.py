"""
Transcription Service
=====================
Flat, simple audio transcription. Tries cloud APIs in priority order,
falls back to local Whisper. No unnecessary classes.

Priority:
  1. Groq  whisper-large-v3
  2. Groq  whisper-large-v3-turbo
  3. Deepgram  nova-2  (whisper-large for Nepali)
  4. Local  faster-whisper
"""

import os
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Suppress duplicate-OpenMP-library crash and force thread counts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
# Force faster-whisper's internal huggingface_hub to use local cache ONLY by default.
# Respect the HF_HUB_OFFLINE environment variable if set (e.g. set to '0' in .env for first-time downloads).
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".weba", ".flac"}

LANGUAGE_CODE_MAP = {
    "nepali": "ne", "tamang": "ne", "newari": "ne",
    "tamang/newari": "ne", "hindi": "hi", "english": "en",
}

class TranscriptionError(Exception):
    """Raised when transcription fails."""


class TranscriptionService:
    """
    Tries transcription models in priority order.
    Just API calls — no provider classes needed.
    """

    # Models to try, in order. Each entry: (provider, model_name)
    MODEL_PRIORITY = [
        ("groq", "whisper-large-v3"),
        ("groq", "whisper-large-v3-turbo"),
        ("deepgram", "nova-2"),
        ("local", "whisper"),
    ]

    def __init__(self, model_size: str = "base", device: str = None, compute_type: str = None):
        self.model_size = model_size
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self._local_model = None  # lazy-loaded

        # API keys
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.deepgram_key = os.getenv("DEEPGRAM_API_KEY")

        # Show what's available at startup
        available = []
        if self.groq_key:
            available.append("Groq")
            print(f"  ✓ Groq API key loaded ({self.groq_key[:8]}...)")
        else:
            print("  ✗ Groq API key NOT found in env")
        if self.deepgram_key:
            available.append("Deepgram")
            print(f"  ✓ Deepgram API key loaded ({self.deepgram_key[:8]}...)")
        else:
            print("  ✗ Deepgram API key NOT found in env")
        available.append("Local Whisper")
        print(f"--- Transcription engines available: {', '.join(available)} ---")

    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_local_model(self):
        """Lazy-load the local faster-whisper model. Just a pip library — no downloads."""
        if self._local_model is not None:
            return
        from faster_whisper import WhisperModel
        threads = os.cpu_count() or 4
        logger.info("Loading faster-whisper model '%s' on %s with %d threads...", self.model_size, self.device, threads)
        self._local_model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type,
            cpu_threads=threads,
            num_workers=1,
        )

    def _normalize_audio(self, audio_path: str) -> str:
        """Normalize any audio format to 16kHz mono WAV."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=os.path.dirname(audio_path))
            audio.export(tmp.name, format="wav")
            tmp.close()
            return tmp.name
        except Exception as e:
            raise TranscriptionError(f"Audio normalization failed: {e}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str, source_language: str = "Nepali") -> Dict[str, Any]:
        t0 = time.time()
        normalized_path = None
        try:
            normalized_path = self._normalize_audio(audio_path)
            lang_code = LANGUAGE_CODE_MAP.get(source_language.lower(), "ne")

            # Try each model in priority order
            last_error = None
            for provider, model in self.MODEL_PRIORITY:
                try:
                    if provider == "groq" and self.groq_key:
                        print(f"  → Trying {provider}/{model}...")
                        result = self._transcribe_groq(normalized_path, lang_code, model)
                    elif provider == "deepgram" and self.deepgram_key:
                        print(f"  → Trying {provider}/{model}...")
                        result = self._transcribe_deepgram(normalized_path, lang_code)
                    elif provider == "local":
                        print(f"  → Trying {provider}/{model}...")
                        result = self._transcribe_local(normalized_path, lang_code)
                    else:
                        print(f"  ⊘ Skipping {provider}/{model} (no API key)")
                        continue

                    result["processing_seconds"] = round(time.time() - t0, 2)
                    result["model_used"] = f"{provider}/{model}"
                    print(f"  ✓ Success via {provider}/{model}")
                    logger.info("Transcribed via %s/%s: %s...", provider, model,
                                result["transcribed_text"][:50])
                    return result

                except Exception as e:
                    last_error = e
                    print(f"  ✗ {provider}/{model} FAILED: {e}")
                    logger.warning("Model %s/%s failed: %s — trying next", provider, model, e)
                    continue

            # All models failed
            raise TranscriptionError(f"All transcription models failed. Last error: {last_error}")
        finally:
            if normalized_path and os.path.exists(normalized_path):
                os.unlink(normalized_path)

    # ------------------------------------------------------------------
    # API calls — just plain functions, no classes
    # ------------------------------------------------------------------
    def _transcribe_groq(self, path: str, lang: str, model: str) -> Dict[str, Any]:
        import httpx
        with open(path, "rb") as f:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.groq_key}"},
                files={"file": (Path(path).name, f)},
                data={"model": model, "language": lang, "response_format": "verbose_json"},
                timeout=60.0,
            )
        if resp.status_code != 200:
            raise TranscriptionError(f"Groq ({model}) error: {resp.text}")
        data = resp.json()
        return {
            "transcribed_text": data.get("text", ""),
            "language_detected": lang,
            "audio_duration_seconds": data.get("duration", 0),
            "segments": data.get("segments", []),
        }

    def _transcribe_deepgram(self, path: str, lang: str) -> Dict[str, Any]:
        import httpx
        # Deepgram nova-2 doesn't support Nepali — use whisper-large for it
        model = "whisper-large" if lang == "ne" else "nova-2"
        url = f"https://api.deepgram.com/v1/listen?model={model}&language={lang}&smart_format=true"
        with open(path, "rb") as f:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Token {self.deepgram_key}", "Content-Type": "audio/wav"},
                content=f.read(), timeout=60.0,
            )
        if resp.status_code != 200:
            raise TranscriptionError(f"Deepgram ({model}) error: {resp.text}")
        data = resp.json()

        # Safe extraction from Deepgram's nested response
        transcript = ""
        try:
            ch = data.get("results", {}).get("channels", [])
            if ch:
                alt = ch[0].get("alternatives", [])
                if alt:
                    transcript = alt[0].get("transcript", "")
        except (KeyError, IndexError):
            pass

        return {
            "transcribed_text": transcript,
            "language_detected": lang,
            "audio_duration_seconds": data.get("metadata", {}).get("duration", 0),
            "segments": [],
        }

    def _transcribe_local(self, path: str, lang: str) -> Dict[str, Any]:
        self._load_local_model()
        segs, info = self._local_model.transcribe(
            path, language=lang, beam_size=1,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
        )
        segments = []
        text_parts = []
        for s in segs:
            segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
            text_parts.append(s.text.strip())
        return {
            "transcribed_text": " ".join(text_parts),
            "language_detected": info.language,
            "audio_duration_seconds": round(info.duration, 2),
            "segments": segments,
        }
