"""
OCR Engine Module
=================
Provides robust OCR text extraction from images and PDF documents
using Tesseract OCR. Supports Nepali, Hindi and English languages by default.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract as pt
from PIL import Image
from ocr.preprocessing import preprocess_pil_image

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

class OCRError(Exception):
    """Raised when an OCR operation fails."""


class OCREngine:
    """
    A configurable OCR engine that extracts text from images and PDFs.

    Parameters
    ----------
    lang : str
        Tesseract language string (e.g. ``"nep+hin+eng"``).
    tesseract_config : str
        Extra Tesseract CLI flags (e.g. ``"--psm 6"``).
    poppler_path : str | None
        Path to Poppler binaries on Windows. If *None*, Poppler must be on
        the system PATH.
    """

    def __init__(
        self,
        lang: str = "nep+hin+eng",
        tesseract_config: str = "--psm 3",
        poppler_path: Optional[str] = None,
    ):
        self.lang = lang
        self.tesseract_config = tesseract_config
        self.poppler_path = poppler_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, file_path: str) -> str:
        """
        Extract text from an image or PDF file.

        Parameters
        ----------
        file_path : str
            Path to an image or PDF file.

        Returns
        -------
        str
            Extracted text.

        Raises
        ------
        OCRError
            If the file does not exist, has an unsupported extension, or
            an error occurs during OCR / PDF conversion.
        """
        path = Path(file_path)

        if not path.exists():
            raise OCRError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise OCRError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if ext in SUPPORTED_PDF_EXTENSIONS:
            logger.info("Processing PDF: %s", file_path)
            return self._process_pdf(str(path))
        else:
            logger.info("Processing image: %s", file_path)
            return self._extract_from_image(str(path))

    # ------------------------------------------------------------------
    # Image OCR
    # ------------------------------------------------------------------
    def _extract_from_image(self, image_input) -> str:
        """
        Run Tesseract on a single image.

        Parameters
        ----------
        image_input : str | numpy.ndarray | PIL.Image.Image
            File path, OpenCV array, or PIL Image.

        Returns
        -------
        str
            Extracted text, stripped of leading/trailing whitespace.
        """
        try:
            if isinstance(image_input, str):
                # file path — read with OpenCV
                img = cv2.imread(image_input)
                if img is None:
                    raise OCRError(
                        f"Failed to read image file: {image_input}"
                    )
            elif isinstance(image_input, np.ndarray):
                img = image_input
            elif isinstance(image_input, Image.Image):
                img = np.array(image_input)
            else:
                raise OCRError(
                    f"Unsupported image input type: {type(image_input).__name__}"
                )

            text: str = pt.image_to_string(
                img,
                lang=self.lang,
                config=self.tesseract_config,
            )
            logger.debug("Extracted %d characters", len(text.strip()))
            return text.strip()

        except pt.TesseractError as exc:
            raise OCRError(f"Tesseract OCR failed: {exc}") from exc

    # ------------------------------------------------------------------
    # PDF → images → OCR
    # ------------------------------------------------------------------
    def _convert_pdf_to_images(self, pdf_path: str) -> list:
        """
        Convert every page of a PDF to a PIL Image.

        Returns
        -------
        list[PIL.Image.Image]
        """
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise OCRError(
                "pdf2image is required for PDF processing. "
                "Install it with: pip install pdf2image"
            ) from exc

        try:
            kwargs = {"pdf_path": pdf_path}
            if self.poppler_path:
                kwargs["poppler_path"] = self.poppler_path

            images = convert_from_path(**kwargs)
            logger.info(
                "Converted PDF to %d page image(s)", len(images)
            )
            return images

        except Exception as exc:
            raise OCRError(
                f"Failed to convert PDF to images: {exc}"
            ) from exc

    def _process_pdf(self, pdf_path: str) -> str:
        """
        Convert a PDF to images, run OCR on each page, and return the
        concatenated text.
        """
        images = self._convert_pdf_to_images(pdf_path)
        page_texts: list[str] = []

        for page_num, pil_img in enumerate(images, start=1):
            logger.info("Running OCR on page %d / %d", page_num, len(images))
            preprocessed = preprocess_pil_image(pil_img)
            text = self._extract_from_image(preprocessed)
            page_texts.append(text)

        combined = "\n\n".join(page_texts)
        logger.info(
            "PDF OCR complete — %d pages, %d total characters",
            len(images),
            len(combined),
        )
        return combined


# ---------------------------------------------------------------------------
# Backward-compatible convenience function
# ---------------------------------------------------------------------------
def run_ocr(image_path) -> str:
    """
    Extract text from an image or PDF.

    This is a thin wrapper around :class:`OCREngine` kept for backward
    compatibility with existing callers.
    """
    engine = OCREngine()
    return engine.process(image_path)