"""
OCR Engine Module
=================
Provides robust OCR text extraction from images and PDF documents using Tesseract.

Strategy:
  - Born-digital PDFs  → PyMuPDF direct extraction (fastest, lossless)
  - Scanned/image PDFs → Tesseract (Optimized for Nepali/Devanagari)
  - Standalone images  → Tesseract (Optimized for Nepali/Devanagari)
  - Word documents     → python-docx / docx2txt
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List

import cv2  # type: ignore
import numpy as np
import pytesseract as pt # type: ignore
import fitz  # type: ignore # PyMuPDF
from PIL import Image
from docx import Document # type: ignore
from ocr.preprocessing import preprocess_pil_image, resize_image # type: ignore

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_WORD_EXTENSIONS = {".docx", ".doc"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS | SUPPORTED_WORD_EXTENSIONS


class OCRError(Exception):
    """Raised when an OCR operation fails."""


class OCREngine:
    """
    A high-precision Tesseract-based engine for Nepali text extraction.

    Parameters
    ----------
    lang : str
        Tesseract language string (e.g. "nep+eng" for Devanagari).
    tesseract_config : str
        Tesseract configuration flags (default: "--psm 3").
    poppler_path : str | None
        Optional path to Poppler binaries (Windows focus).
    """

    def __init__(
        self,
        lang: str = "nep+eng",
        tesseract_config: str = "--psm 3",
        poppler_path: Optional[str] = None,
    ):
        self.lang = lang
        self.tesseract_config = tesseract_config
        self.poppler_path = poppler_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, file_path: str) -> List[str]:
        """Extract text from supported files, returning one string per page."""
        path = Path(file_path)
        if not path.exists():
            raise OCRError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise OCRError(f"Unsupported file type '{ext}'.")

        if ext in SUPPORTED_PDF_EXTENSIONS:
            # 1. Try direct extraction (born-digital)
            direct_text = self._process_pdf_direct(str(path))
            if direct_text and any(page.strip() for page in direct_text):
                return direct_text

            # 2. Scanned OCR fallback
            return self._process_pdf_ocr(str(path), return_list=True)

        elif ext in SUPPORTED_WORD_EXTENSIONS:
            return self._process_word(str(path))

        else:
            return [self._extract_from_image(str(path))]

    # ------------------------------------------------------------------
    # File Specific Extractors
    # ------------------------------------------------------------------
    def _process_word(self, file_path: str) -> List[str]:
        try:
            path = Path(file_path)
            if path.suffix.lower() == ".docx":
                doc = Document(file_path)
                return ["\n".join([p.text for p in doc.paragraphs]).strip()]
            else:
                import docx2txt # type: ignore
                return [docx2txt.process(file_path).strip()]
        except Exception as e:
            raise OCRError(f"Word extraction failed: {e}")

    def _extract_from_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> str:
        """
        Clean, normalise, and OCR a single image using Tesseract.
        """
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
            elif isinstance(image_input, np.ndarray):
                img = image_input
            elif isinstance(image_input, Image.Image):
                img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                raise OCRError("Unsupported image input type.")
            
            if img is None:
                raise OCRError("Could not load image.")

            # Ensure optimal size for Tesseract (max 4000px)
            img = resize_image(img, max_dim=4000)

            return self._tesseract_ocr(img)
        except Exception as exc:
            logger.error(f"OCR failed for image: {exc}")
            return ""

    def _tesseract_ocr(self, img: np.ndarray) -> str:
        """
        High-precision Tesseract pipeline including denoising and contrast boost.
        """
        try:
            # 1. Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # 2. Denoising (Vital for phone photos / noisy scans)
            # Parameters h=10, searchWindow=21 are good for document grain
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # 3. High-Contrast Binarization (Otsu's method)
            # Tesseract loves clean black-on-white text
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 4. Add a small white border (Padding)
            # Helps Tesseract when text is too close to edges
            binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

            # 5. Extract
            text = pt.image_to_string(
                binary,
                lang=self.lang,
                config=self.tesseract_config
            )
            return text.strip()
        except Exception as exc:
            logger.warning(f"Tesseract inference error: {exc}")
            return ""

    # ------------------------------------------------------------------
    # PDF Logic
    # ------------------------------------------------------------------
    def _process_pdf_direct(self, pdf_path: str) -> List[str]:
        try:
            doc = fitz.open(pdf_path)
            return [page.get_text("text").strip() for page in doc]
        except:
            return []

    def _process_pdf_ocr(self, pdf_path: str, return_list: bool = False) -> Union[str, List[str]]:
        from concurrent.futures import ThreadPoolExecutor
        from pdf2image import convert_from_path # type: ignore

        try:
            images = convert_from_path(pdf_path, dpi=300, thread_count=4, poppler_path=self.poppler_path)
            
            def ocr_page(args):
                page_num, pil_img = args
                # Use standard grayscale preprocessing for Tesseract
                # We do the denoising inside _tesseract_ocr
                text = self._extract_from_image(pil_img)
                return page_num, text

            with ThreadPoolExecutor(max_workers=min(len(images), 8)) as executor:
                results = list(executor.map(ocr_page, enumerate(images, start=1)))

            results.sort(key=lambda x: x[0])
            texts = [r[1] for r in results]

            return texts if return_list else "\n\n".join(texts)
        except Exception as exc:
            raise OCRError(f"PDF OCR failed: {exc}")


def run_ocr(path: str) -> str:
    """Backward-compatible entry point."""
    engine = OCREngine()
    res = engine.process(path)
    return res[0] if isinstance(res, list) else res