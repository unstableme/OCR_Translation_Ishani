import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables at the very beginning
load_dotenv(find_dotenv(), override=False)


from fastapi import FastAPI, UploadFile, File, HTTPException
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

from db.tables import Base, Document, OCRResult, Translation
from ocr.preprocessing import preprocess_image
from ocr.ocr_engine import OCREngine, OCRError, SUPPORTED_EXTENSIONS
from ocr.translator import translate_text, detect_language

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

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
    source_lang: str = "Tamang",
    target_lang: str = "Nepali"
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