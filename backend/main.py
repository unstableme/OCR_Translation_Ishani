import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables at the very beginning
load_dotenv(find_dotenv(), override=True)

# Configure logging early to catch initialization logs
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
)

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel
from db.connection import engine, SessionLocal

from typing import Union, List

class TranslationRequest(BaseModel):
    text: Union[str, List[str]]
    source_lang: str = "Tamang" # Backend supports "Tamang", "Newari", or combinations like "Tamang/Newari"
    target_lang: str = "Nepali"

# Quick diagnostic for database connection
db_url = os.getenv("DATABASE_URL", "")
logger = logging.getLogger(__name__)
# Mask the password in logs
safe_url = db_url.split('@')[-1] if '@' in db_url else db_url
print(f"DEBUG: Database connecting to -> {safe_url}")

from db.tables import Base, Document, OCRResult, Translation, AudioTranscription
from ocr.preprocessing import preprocess_image
from ocr.ocr_engine import OCREngine, OCRError, SUPPORTED_EXTENSIONS
from ocr.translator import translate_text, detect_language
from audio.transcription_service import (
    TranscriptionService,
    TranscriptionError,
    SUPPORTED_AUDIO_EXTENSIONS,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Create database tables on startup
#Base.metadata.create_all(bind=engine)


# Shared OCR engine instance (now uses Hybrid under the hood)
ocr_engine = OCREngine()

# Shared Transcription engine instance (lazy-loads Whisper model on first call)
model_size = os.getenv("WHISPER_MODEL", "tiny")
transcription_engine = TranscriptionService(model_size=model_size)





@app.get("/")
async def root():
    """Health check / status endpoint."""
    return {"status": "running", "message": "OCR & Translation API is live"}


@app.post("/translate")
async def translate_only(request: TranslationRequest):
    """
    Directly translate text strings without OCR or file upload.
    Used for UI-based direct text input or table row translations.
    """
    import time
    t0 = time.time()

    if isinstance(request.text, str):
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text for translation cannot be empty")
    elif isinstance(request.text, list):
        if not any(t.strip() for t in request.text if isinstance(t, str)):
            raise HTTPException(status_code=400, detail="Text list for translation cannot be empty")
    else:
         raise HTTPException(status_code=400, detail="Invalid text format. Expected string or list of strings.")

    try:
        translated_text, model_used = translate_text(
            request.text, 
            request.source_lang, 
            request.target_lang
        )
        total_duration = time.time() - t0
        return {
            "translated_text": translated_text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "model_used": model_used,
            "timing": {
                "llm_api_response_seconds": round(total_duration, 2),
                "total_processing_seconds": round(total_duration, 2)
            }
        }
    except Exception as e:
        logger.error(f"Direct translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_language")
async def language_detection_endpoint(file: UploadFile = File(...)):
    """
    Detect the language of an uploaded document (image or PDF).
    Uses professional OCR and LLM-based identification for Himalayan languages.
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    import time
    t0 = time.time()
    
    try:
        # 1. Save File (Temporary)
        os.makedirs("uploads", exist_ok=True)
        temp_filename = f"detect_{int(t0)}_{filename}"
        file_path = f"uploads/{temp_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # 2. OCR Extraction (Focusing on first page for speed/efficiency)
        extracted_pages = ocr_engine.process(file_path)
        if not extracted_pages:
            return {
                "message": "No text extracted from document",
                "language": "Unknown",
                "language_code": "unknown",
                "confidence": 0.0
            }
            
        first_page_text = extracted_pages[0]
        
        # 3. Language Identification (LLM based snippet analysis)
        detection_result = detect_language(first_page_text)
        
        duration = time.time() - t0
        
        return {
            "message": "Language identified successfully",
            "filename": filename,
            "language": detection_result.get("language", "Unknown"),
            "language_code": detection_result.get("code", "unknown"),
            "confidence": detection_result.get("confidence", 0.0),
            "snippet": first_page_text[:120].strip() + "...",
            "timing": {
                "total_processing_seconds": round(duration, 2)
            }
        }
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    source_lang: str = Form("Tamang"),
    target_lang: str = Form("Nepali")
):
    """
    Upload a document (image or PDF) for OCR and translation.
    """
    # --- validate file type ---
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    import time
    t0 = time.time()  # Start of request
    
    db = SessionLocal()
    try:
        # --- 1. File Upload / Save ---
        t_upload_start = time.time()
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{filename}"

        # Optimization: Use a single read and write
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        t_upload_end = time.time()
        upload_duration = t_upload_end - t_upload_start

        # Create Document object but defer final commit to reduce overhead
        doc = Document(
            original_filename=filename,
            stored_path=file_path,
            status="Processing",
        )
        db.add(doc)
        db.flush() # Get the ID without full commit yet
        doc_id = doc.id
        
        t_db_init_end = time.time()
        db_init_duration = t_db_init_end - t_upload_end

        # --- 2. OCR Processing ---
        t_ocr_start = time.time()
        # Detailed result includes text, confidence, and bounding boxes
        detailed_result = ocr_engine.process_detailed(file_path)
        extracted_pages = [p["text"] for p in detailed_result["pages"]]
        extracted_text = "\n\n".join(extracted_pages)
        avg_confidence = (
            sum(p["confidence"] for p in detailed_result["pages"])
            / len(detailed_result["pages"])
            if detailed_result["pages"] else 0.0
        )
        
        t_ocr_end = time.time()
        ocr_duration = t_ocr_end - t_ocr_start

        # --- 3. LLM API Response ---
        t_llm_start = time.time()
        # Passing the list of pages triggers parallel translation in the translator module
        translated_text, model_used = translate_text(extracted_pages, source_lang, target_lang)
        t_llm_end = time.time()
        llm_duration = t_llm_end - t_llm_start

        # --- 4. Final DB Updates (Consolidated) ---
        t_db_final_start = time.time()
        
        ocr_result = OCRResult(
            document_id=doc_id,
            extracted_text=extracted_text,
            status="Extracted",
        )
        db.add(ocr_result)

        translated_result = Translation(
            document_id=doc_id,
            translated_text=translated_text,
            model_used=model_used,
            status="Completed",
        )
        db.add(translated_result)

        doc.status = "Completed"
        db.commit() # Single commit for all results
        
        t_db_final_end = time.time()
        db_final_duration = t_db_final_end - t_db_final_start

        total_duration = time.time() - t0
        
        # Log detailed Telemetry
        telemetry = (
            f"TELEMETRY: Total={total_duration:.2f}s | "
            f"Upload={upload_duration:.2f}s | "
            f"DB_Init={db_init_duration:.2f}s | "
            f"OCR={ocr_duration:.2f}s | "
            f"LLM={llm_duration:.2f}s | "
            f"DB_Final={db_final_duration:.2f}s"
        )
        logger.info(telemetry)
        print(telemetry) # Ensure it's visible in terminal

        return {
            "message": "Document processed successfully",
            "document_id": doc_id,
            "ocr_result_id": ocr_result.id,
            "translation_id": translated_result.id,
            "extracted_text": extracted_text,
            "translated_text": translated_text,
            "file_url": f"/uploads/{filename}",
            "ocr_confidence": round(avg_confidence, 4),
            "ocr_pages": detailed_result["pages"],
            "ocr_strategy": detailed_result.get("ocr_strategy", "unknown"),
            "debug_image_urls": detailed_result.get("debug_images", []),
            "timing": {
                "file_upload_seconds": round(upload_duration, 2),
                "db_init_seconds": round(db_init_duration, 2),
                "ocr_processing_seconds": round(ocr_duration, 2),
                "llm_api_response_seconds": round(llm_duration, 2),
                "db_final_seconds": round(db_final_duration, 2),
                "total_processing_seconds": round(total_duration, 2)
            }
        }
    except OCRError as e:
        logger.error("OCR failed: %s", e)
        db.rollback()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback

        traceback.print_exc()
        db.rollback()
        return {"error": str(e), "details": traceback.format_exc()}
    finally:
        db.close()


@app.post("/upload_audio")
async def upload_audio(
    file: UploadFile = File(...),
    source_lang: str = Form("Nepali"),
    target_lang: str = Form("Nepali"),
):
    """
    Upload an audio file for transcription and translation.
    
    Pipeline: Audio File → Whisper Transcription → raw text → LLM Translation
    Mirrors the /upload endpoint flow but replaces OCR with speech-to-text.
    
    Supported formats: .mp3, .wav, .m4a, .ogg, .webm, .weba, .flac
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format '{ext}'. "
                f"Accepted: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
            ),
        )

    import time
    t0 = time.time()

    db = SessionLocal()
    try:
        # 1. Save Audio File
        t_upload_start = time.time()
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{filename}"

        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)

        t_upload_end = time.time()
        upload_duration = t_upload_end - t_upload_start

        # Create Document record
        doc = Document(
            original_filename=filename,
            stored_path=file_path,
            status="Processing",
        )
        db.add(doc)
        db.flush() #pushes data to db but doesn't save permanently(still in temporary state)
        doc_id = doc.id

        t_db_init_end = time.time()
        db_init_duration = t_db_init_end - t_upload_end

        # 2. Audio Transcription 
        t_transcribe_start = time.time()
        transcription_result = transcription_engine.transcribe(
            file_path, source_language=source_lang
        )
        extracted_text = transcription_result["transcribed_text"]

        t_transcribe_end = time.time()
        transcribe_duration = t_transcribe_end - t_transcribe_start

        # 3. LLM Translation
        t_llm_start = time.time()
        translated_text, model_used = translate_text(
            extracted_text, source_lang, target_lang
        )
        t_llm_end = time.time()
        llm_duration = t_llm_end - t_llm_start

        # 4. DB Persistence 
        t_db_final_start = time.time()

        audio_record = AudioTranscription(
            document_id=doc_id,
            transcribed_text=extracted_text,
            language_detected=transcription_result.get("language_detected", ""),
            audio_duration=transcription_result.get("audio_duration_seconds", 0.0),
            status="Transcribed",
        )
        db.add(audio_record)

        translated_result = Translation(
            document_id=doc_id,
            translated_text=translated_text,
            model_used=model_used,
            status="Completed",
        )
        db.add(translated_result)

        doc.status = "Completed"
        db.commit()

        t_db_final_end = time.time()
        db_final_duration = t_db_final_end - t_db_final_start

        total_duration = time.time() - t0

        telemetry = (
            f"TELEMETRY [AUDIO]: Total={total_duration:.2f}s | "
            f"Upload={upload_duration:.2f}s | "
            f"DB_Init={db_init_duration:.2f}s | "
            f"Transcription={transcribe_duration:.2f}s | "
            f"LLM={llm_duration:.2f}s | "
            f"DB_Final={db_final_duration:.2f}s"
        )
        logger.info(telemetry)

        return {
            "message": "Audio processed successfully",
            "document_id": doc_id,
            "audio_transcription_id": audio_record.id,
            "translation_id": translated_result.id,
            "extracted_text": extracted_text,
            "translated_text": translated_text,
            "language_detected": transcription_result.get("language_detected", ""),
            "audio_duration_seconds": transcription_result.get("audio_duration_seconds", 0),
            "segments": transcription_result.get("segments", []),
            "timing": {
                "file_upload_seconds": round(upload_duration, 2),
                "db_init_seconds": round(db_init_duration, 2),
                "transcription_seconds": round(transcribe_duration, 2),
                "llm_api_response_seconds": round(llm_duration, 2),
                "db_final_seconds": round(db_final_duration, 2),
                "total_processing_seconds": round(total_duration, 2),
            },
        }
    except TranscriptionError as e:
        logger.error("Transcription failed: %s", e)
        db.rollback()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/transcribe")
async def transcribe_audio_only(
    file: UploadFile = File(...),
    source_lang: str = Form("Nepali"),
    force_model: str = Form(None),
):
    """
    Transcribe an audio file WITHOUT translation.
    Used for live recordings where user may want to review text before translating.
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format '{ext}'. "
                f"Accepted: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
            ),
        )

    import time
    t0 = time.time()

    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{filename}"

        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)

        result = transcription_engine.transcribe(
            file_path, source_language=source_lang, force_model=force_model
        )

        duration = time.time() - t0

        return {
            "message": "Audio transcribed successfully",
            "transcribed_text": result["transcribed_text"],
            "language_detected": result.get("language_detected", ""),
            "audio_duration_seconds": result.get("audio_duration_seconds", 0),
            "segments": result.get("segments", []),
            "timing": {
                "total_processing_seconds": round(duration, 2),
            },
        }
    except TranscriptionError as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Transcription error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/transcribe")
async def ws_live_transcribe(websocket: WebSocket):
    """
    Live streaming transcription via WebSocket.

    Protocol:
      CLIENT → SERVER:
        - Binary message: raw audio chunk bytes (webm/opus from MediaRecorder)
        - Text message "done": signals recording has ended

      SERVER → CLIENT:
        - {"type": "segment", "text": "...", "chunk_index": N}  — new transcribed text
        - {"type": "done"}                                        — all done
        - {"type": "error", "message": "..."}                    — error occurred
        - {"type": "status", "message": "..."}                   — info/debug messages

    The client should accumulate all "segment" texts to build the full transcript.
    """
    import json
    import asyncio
    import tempfile

    await websocket.accept()
    logger.info("WS /ws/transcribe: connection accepted")

    # Pull language hint from query param (e.g. ?lang=Nepali)
    source_language = websocket.query_params.get("lang", "Nepali")
    # Pull optional forced model parameter (e.g. ?model=groq/whisper-large-v3)
    force_model = websocket.query_params.get("model", None)

    chunk_index = 0
    audio_chunks: list[bytes] = []   # accumulate all audio for the session

    try:
        # Pre-load local whisper model in a background thread
        # so it doesn't block the event loop and kill the WebSocket
        # Only pre-load if we are in auto mode or local model is explicitly forced
        if (force_model is None or force_model == "local") and hasattr(transcription_engine, '_load_local_model'):
            await asyncio.to_thread(transcription_engine._load_local_model)
        
        while True:
            # Receive next message — could be bytes (audio) or text ("done")
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Connection timed out after 60 seconds of inactivity"
                }))
                break

            # --- Text message: only "done" is expected ---
            if message["type"] == "websocket.receive" and message.get("text") is not None:
                text_msg = message["text"]
                if text_msg == "done":
                    logger.info("WS /ws/transcribe: client signalled 'done'")
                    await websocket.send_text(json.dumps({"type": "done"}))
                    break
                # Ignore any other text messages
                continue

            # --- Binary message: audio chunk ---
            if message["type"] == "websocket.receive" and message.get("bytes") is not None:
                audio_data: bytes = message["bytes"]

                if len(audio_data) < 100:
                    # Too small to be useful audio, skip silently
                    continue

                # Accumulate chunk
                audio_chunks.append(audio_data)
                chunk_index += 1

                # Write all accumulated audio so far to a temp webm file
                # This gives Whisper enough context for accurate transcription
                tmp_webm = None
                try:
                    os.makedirs("uploads/ws_temp", exist_ok=True)

                    # Write accumulated bytes as a single webm file
                    with tempfile.NamedTemporaryFile(
                        suffix=".webm", delete=False, dir="uploads/ws_temp"
                    ) as f:
                        for chunk in audio_chunks:
                            f.write(chunk)
                        tmp_webm = f.name

                    # Run transcription in a background thread so the event loop
                    # stays alive for WebSocket heartbeats and message handling
                    result = await asyncio.to_thread(
                        transcription_engine.transcribe,
                        tmp_webm,
                        source_language,
                        force_model=force_model,
                    )
                    full_text = result["transcribed_text"]
                    model_used = result.get("model_used")

                    if websocket.client_state == WebSocketState.CONNECTED:
                        if full_text:
                            await websocket.send_text(json.dumps({
                                "type": "segment",
                                "text": full_text,
                                "chunk_index": chunk_index,
                                "is_final": False,
                                "model_used": model_used,
                            }))
                            logger.info(
                                "WS chunk %d transcribed: %d chars (%s)", chunk_index, len(full_text), model_used
                            )
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": "No speech detected in this chunk yet..."
                            }))
                    else:
                        logger.warning("WS client disconnected during transcription — skipping send")
                        break

                except Exception as exc:
                    logger.error("WS transcription chunk error: %s", exc)
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": f"Transcription error: {str(exc)}"
                            }))
                        except Exception:
                            pass
                    else:
                        break
                finally:
                    # Clean up temp files
                    if tmp_webm and os.path.exists(tmp_webm):
                        try:
                            os.unlink(tmp_webm)
                        except OSError:
                            pass

            # Handle disconnect message
            elif message["type"] == "websocket.disconnect":
                logger.info("WS /ws/transcribe: client disconnected")
                break

    except WebSocketDisconnect:
        logger.info("WS /ws/transcribe: WebSocketDisconnect")
    except Exception as e:
        if "once a close message has been sent" in str(e):
            logger.info("WS /ws/transcribe: connection was closed by the client during processing")
        else:
            logger.error("WS /ws/transcribe unexpected error: %s", e)
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                except Exception:
                    pass
    finally:
        logger.info("WS /ws/transcribe: connection closed")
