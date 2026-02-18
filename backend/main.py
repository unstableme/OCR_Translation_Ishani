import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables at the very beginning
load_dotenv(find_dotenv())

from fastapi import FastAPI, UploadFile, File, HTTPException
from db.connection import engine, SessionLocal
from db.tables import Base, Document, OCRResult, Translation
from ocr.preprocessing import preprocess_image
from ocr.ocr_engine import OCREngine, OCRError, SUPPORTED_EXTENSIONS
from ocr.translator import translate_to_nepali

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

# Shared OCR engine instance
ocr_engine = OCREngine()


@app.get("/")
async def root():
    """Health check / status endpoint."""
    return {"status": "running", "message": "OCR & Translation API is live"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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

        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        t_upload_end = time.time()
        upload_duration = t_upload_end - t_upload_start

        doc = Document(
            original_filename=filename,
            stored_path=file_path,
            status="Processing",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # --- 2. OCR Processing ---
        t_ocr_start = time.time()
        # process now returns a list of strings (one per page)
        extracted_pages = ocr_engine.process(file_path)
        extracted_text = "\n\n".join(extracted_pages)
        
        t_ocr_end = time.time()
        ocr_duration = t_ocr_end - t_ocr_start

        # --- 3. LLM API Response ---
        t_llm_start = time.time()
        # Passing the list of pages triggers parallel translation in the translator module
        translated_text, model_used = translate_to_nepali(extracted_pages)
        t_llm_end = time.time()
        llm_duration = t_llm_end - t_llm_start

        # --- DB Updates ---
        ocr_result = OCRResult(
            document_id=doc.id,
            extracted_text=extracted_text,
            status="Extracted",
        )
        db.add(ocr_result)
        db.commit()
        db.refresh(ocr_result)

        translated_result = Translation(
            document_id=doc.id,
            translated_text=translated_text,
            model_used=model_used,
            status="Completed",
        )
        db.add(translated_result)
        db.commit()
        db.refresh(translated_result)

        doc.status = "Completed"
        db.commit()

        total_duration = time.time() - t0
        
        # Log results
        logger.info(f"PERFORMANCE: Total={total_duration:.2f}s | Upload={upload_duration:.2f}s | OCR={ocr_duration:.2f}s | LLM={llm_duration:.2f}s")

        return {
            "message": "Document processed successfully",
            "document_id": doc.id,
            "ocr_result_id": ocr_result.id,
            "translation_id": translated_result.id,
            "extracted_text": extracted_text,
            "translated_text": translated_text,
            "file_url": f"/uploads/{filename}",
            "timing": {
                "file_upload_seconds": round(upload_duration, 2),
                "ocr_processing_seconds": round(ocr_duration, 2),
                "llm_api_response_seconds": round(llm_duration, 2),
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