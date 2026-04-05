"""
OCR Engine Module
=================

A production-grade OCR extraction pipeline optimized for Himalayan languages (Nepali, Tamang, Newari).
The module provides a multi-layered architecture that balances speed and accuracy through an
intelligent fallback mechanism.

### Engine Hierarchy:
1. **OCREngine (Wrapper)**: The main entry point used by `main.py`. Handles file routing (PDF/Image/Word)
   and manages the High-Level `HybridOCREngine`.
2. **HybridOCREngine**: The "brain" of the system. Implements a two-stage strategy:
   - **Fast Path**: Multi-PSM Tesseract for quick, standard document extraction.
   - **Slow Path**: docTR for layout analysis + Parallel Tesseract on crops for complex cases.
3. **Core Engines**:
   - `TesseractOCREngine`: Best-in-class for Devanagari script via the `nep` language pack.
   - `DocTROCREngine`: State-of-the-art Deep Learning models for robust layout and boundary detection.

"""
# pyre-ignore-all-errors
import os
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# Suppress OpenMP/Threading warnings from PyTorch/OpenCV conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pytesseract as pt
import fitz  # PyMuPDF
from docx import Document
from PIL import Image as PILImage

from ocr.preprocessing import (
    preprocess_pil_image,
    preprocess_image,
    preprocess_array,
    DEFAULT_CONFIG as PREPROCESS_DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_WORD_EXTENSIONS = {".docx", ".doc"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS | SUPPORTED_WORD_EXTENSIONS


class OCRError(Exception):
    """Raised when an OCR operation fails."""


# ===================================================================
# Result helpers
# ===================================================================

def _make_page_result(text: str, confidence: float = 0.0,
                      boxes: list | None = None) -> dict:
    """Build a single-page result dict."""
    return {
        "text": text,
        "confidence": round(float(confidence), 4),
        "boxes": boxes or [],
    }


def _make_result(pages: list[dict]) -> dict:
    """Wrap page results in the standard output format."""
    return {"pages": pages}


# ===================================================================
# Tesseract Engine
# ===================================================================

class TesseractOCREngine:
    """
    Tesseract-based OCR engine wrapper.
    
    Optimized for Devanagari script documents using the Tesseract `nep` (Nepali) 
    and `eng` (English) language packs. This engine is the primary workhorse 
    for standard document extraction due to its high performance and native 
    script support.
    """

    def __init__(
        self,
        lang: str = "nep+eng",
        tesseract_config: str = "--psm 3",
        preprocess_config: Optional[dict] = None,
    ):
        self.lang = lang
        self.tesseract_config = tesseract_config
        self.preprocess_config = preprocess_config

    def process_image(self, image: np.ndarray) -> dict:
        """
        Run Tesseract OCR on a preprocessed image with structured text reconstruction.

        This method performs a single call to Tesseract's `image_to_data` and 
        reconstructs the original document structure (lines, paragraphs, and blocks) 
        by analyzing the spatial metadata returned for each word.

        Args:
            image (np.ndarray): Preprocessed image (numpy array).

        Returns:
            dict: Standardized result format:
                {
                    "pages": [{
                        "text": str (Reconstructed content with line breaks),
                        "confidence": float (0.0 to 1.0),
                        "boxes": list[dict] (Words with individual bboxes)
                    }]
                }

        Internal Logic:
        1. Calls `pt.image_to_data` to get word-level coordinates and confidence.
        2. Groups words into `(block, paragraph, line)` buckets.
        3. Joins words into lines and separates blocks with double newlines.
        4. Calculates a weighted average confidence based on word-level results.
        """
        try:
            # Single Tesseract call — get everything from image_to_data
            data = pt.image_to_data(
                image, lang=self.lang, config=self.tesseract_config,
                output_type=pt.Output.DICT,
            )

            # Reconstruct structured text preserving line/paragraph breaks
            confidences, boxes = [], []
            lines_by_block = {}  # {(block_num, par_num, line_num): [words]}
            block_nums_seen = []  # Track block ordering

            for i, word_text in enumerate(data["text"]):
                word_text = word_text.strip()
                conf = int(data["conf"][i])
                block_num = data["block_num"][i]
                par_num = data["par_num"][i]
                line_num = data["line_num"][i]

                if word_text and conf > 0:
                    confidences.append(conf / 100.0)
                    boxes.append({
                        "text": word_text,
                        "confidence": round(float(conf / 100.0), 4),
                        "bbox": [
                            data["left"][i],
                            data["top"][i],
                            data["left"][i] + data["width"][i],
                            data["top"][i] + data["height"][i],
                        ],
                    })

                    key = (block_num, par_num, line_num)
                    if key not in lines_by_block:
                        lines_by_block[key] = []
                    lines_by_block[key].append(word_text)

                    if block_num not in block_nums_seen:
                        block_nums_seen.append(block_num)

            # Build structured text: lines joined by space, paragraphs by newline,
            # blocks separated by double-newline
            text_parts = []
            prev_block = None
            for key in sorted(lines_by_block.keys()):
                block_num, par_num, line_num = key
                line_text = " ".join(lines_by_block[key])
                if prev_block is not None and block_num != prev_block:
                    text_parts.append("")  # Double-newline between blocks
                text_parts.append(line_text)
                prev_block = block_num

            full_text = "\n".join(text_parts).strip()
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return _make_result([_make_page_result(full_text, avg_conf, boxes)])

        except pt.TesseractError as exc:
            raise OCRError(f"Tesseract OCR failed: {exc}") from exc


# ===================================================================
# docTR Engine
# ===================================================================

class DocTROCREngine:
    """
    docTR-based OCR and Layout Analysis engine.
    
    Uses a Deep Learning pipeline (db_resnet50 for detection + crnn_vgg16_bn 
    for recognition) to provide robust OCR results. In this system, docTR 
    is primarily leveraged for its superior layout analysis and boundary 
    detection capabilities.
    """

    def __init__(self, preprocess_config: Optional[dict] = None):
        self.preprocess_config = preprocess_config
        self._model = None  # Lazy-loaded on first call

    def _load_model(self):
        """
        Lazy-load the docTR model.
        
        This prevents unnecessary memory usage if docTR is not required 
        (e.g., if the Tesseract 'Fast Path' succeeds). Downloads weights 
        on the very first invocation if they are not cached.
        """
        if self._model is not None:
            return
        try:
            from doctr.models import ocr_predictor
            logger.info("Loading docTR model (db_resnet50 + crnn_vgg16_bn)…")
            self._model = ocr_predictor(
                det_arch="db_resnet50",
                reco_arch="crnn_vgg16_bn",
                pretrained=True,
            )
            logger.info("docTR model loaded successfully")
        except ImportError:
            raise OCRError(
                "python-doctr is not installed. "
                "Install with: pip install python-doctr[torch] torch torchvision"
            )
        except Exception as exc:
            raise OCRError(f"Failed to load docTR model: {exc}") from exc

    def _process_raw(self, image: np.ndarray):
        """Run docTR and return the raw Document object (for internal layout analysis)."""
        self._load_model()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return self._model([image_rgb])

    def get_blocks(self, image: np.ndarray) -> list[dict]:
        """
        Perform Layout Analysis to detect physical text blocks.

        Unlike `process_image`, this method focuses on identifying the 
        high-level structure of the document (paragraphs/blocks) without 
        necessarily relying on docTR for the final text recognition.

        Args:
            image (np.ndarray): Image as a numpy array.

        Returns:
            list[dict]: A list of detected blocks, each containing a bbox 
                and word-level coordinates for masking.
        """
        result = self._process_raw(image)
        blocks = []
        for page in result.pages:
            height, width = page.dimensions
            for block in page.blocks:
                geo = block.geometry # [(x_min, y_min), (x_max, y_max)]
                blocks.append({
                    "bbox": [
                        int(geo[0][0] * width), int(geo[0][1] * height),
                        int(geo[1][0] * width), int(geo[1][1] * height)
                    ],
                    # Store words for masking purposes
                    "words": [
                        [int(word.geometry[0][0] * width), int(word.geometry[0][1] * height),
                         int(word.geometry[1][0] * width), int(word.geometry[1][1] * height)]
                        for line in block.lines for word in line.words
                    ],
                    "confidence": np.mean([word.confidence for line in block.lines for word in line.words]) if block.lines else 0.0
                })
        return blocks

        return blocks



# ===================================================================
# Hybrid Engine (docTR + Tesseract fallback)
# ===================================================================

class HybridOCREngine:
    """
    Intelligent OCR controller that manages Tesseract and docTR.
    
    Implements a "Search and Fallback" strategy to balance extraction 
    quality and processing time.
    
    Processing Strategy:
    --------------------
    1.  **Fast Path**: Tries Tesseract with multiple PSM (Page Segmentation 
        Modes). If high confidence exists, it exits early.
    2.  **Slow Path**: If Tesseract fails, it triggers docTR for layout 
        detection, followed by parallel Tesseract extraction on the detected blocks.
    3.  **Intelligent Masking**: Removes non-text 'sidebar noise' while 
        shielding actual text areas during high-complexity OCR runs.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        lang: str = "nep+eng",
        tesseract_config: str = "--psm 3",
        preprocess_config: Optional[dict] = None,
    ):
        self.threshold = confidence_threshold
        self.doctr = DocTROCREngine(preprocess_config=preprocess_config)
        self.tesseract = TesseractOCREngine(
            lang=lang,
            tesseract_config=tesseract_config,
            preprocess_config=preprocess_config,
        )
        self.preprocess_config = preprocess_config

    def process_image(self, image: np.ndarray) -> dict:
        """
        Execute the primary Hybrid OCR pipeline for an image.

        Args:
            image (np.ndarray): The source image as a numpy array.
            debug_filename (Optional[str]): Filename for saving diagnostic images.

        Returns:
            dict: The final OCR results, including strategy metadata and 
                diagnostic URLs (if the Slow Path was triggered).

        ### Detailed Flow:
        
        #### Phase 1: Fast Path (Multi-PSM Tesseract)
        Tesseract is run sequentially with PSM 4, 6, and 3. The engine 
        constantly evaluates the character count and confidence score.
        - **Success Condition**: > 500 characters and > 85% confidence.
        - **Fallback Trigger**: If ALL modes result in < 100 characters.

        #### Phase 2: Slow Path (docTR Layout + Parallel Tesseract)
        Used for documents with low contrast, handwritten elements, or 
        complex tables where standard Tesseract "blindly" misses blocks.
        - **Layout Analysis**: docTR detects bounding boxes for text blocks.
        - **Sidebar Masking**: Identifies 15% right-sidebar area and protects 
          text while masking white-space artifacts.
        - **Parallelized Extraction**: Spawns multiple threads (limited to 6) 
          to run Tesseract on each individual block crop for maximum accuracy.
        """
        import time
        t_start = time.time()

        # ============================================================
        # FAST PATH: Multi-PSM Tesseract
        # --psm 4: Single column of variable sizes (best for government letters)
        # --psm 6: Uniform block of text (good for body-heavy documents)
        # --psm 3: Fully automatic segmentation (general fallback)
        # ============================================================
        best_result = None
        best_text_len = 0
        best_conf = 0.0
        best_psm = ""

        psm_modes = ["--psm 4", "--psm 6", "--psm 3"]

        for psm in psm_modes:
            engine = TesseractOCREngine(
                lang=self.tesseract.lang,
                tesseract_config=psm,
                preprocess_config=self.preprocess_config,
            )
            result = engine.process_image(image)
            page = result["pages"][0]
            text = page["text"].strip()
            conf = page["confidence"]

            logger.info("[Hybrid] Tried %s: text_len=%d, conf=%.4f", psm, len(text), conf)

            # Pick the mode that extracted the most text
            if len(text) > best_text_len:
                best_result = result
                best_text_len = len(text)
                best_conf = conf
                best_psm = psm

            # Early exit: substantial text with good confidence — no need to try more modes
            if len(text) > 500 and conf >= self.threshold:
                logger.info("[Hybrid] Early exit on %s (text_len=%d > 500, conf=%.4f >= %.2f)",
                            psm, len(text), conf, self.threshold)
                break

        t_fast = time.time() - t_start
        logger.info("[Hybrid] Fast path complete in %.2fs — best_psm=%s, best_text_len=%d, best_conf=%.4f",
                    t_fast, best_psm, best_text_len, best_conf)

        # Accept fast path if Tesseract extracted meaningful content
        # For a full-page document, even partial extraction should give > 100 chars
        if best_text_len > 100:
            logger.info("[Hybrid] Fast path ACCEPTED (text_len=%d > 100, psm=%s)",
                        best_text_len, best_psm)
            best_result["ocr_strategy"] = f"tesseract_multi_psm_{best_psm.replace('--', '')}"
            return best_result

        # ============================================================
        # SLOW PATH: docTR layout detection + parallel Tesseract on crops
        # Only triggered when Tesseract truly fails (< 100 chars extracted)
        # ============================================================
        logger.info("[Hybrid] Fast path rejected (best_text_len=%d < 100). "
                    "Falling back to docTR layout + parallel Tesseract...",
                    best_text_len)

        # --- 1. Layout Analysis via docTR ---
        try:
            blocks = self.doctr.get_blocks(image)
        except Exception as e:
            import traceback
            logger.warning("docTR layout analysis failed: %s\n%s. Returning best Tesseract result.",
                           e, traceback.format_exc())
            if best_result:
                best_result["ocr_strategy"] = "tesseract_multi_psm_doctr_failed"
                return best_result
            return _make_result([_make_page_result("", 0.0)])

        if not blocks:
            logger.info("No blocks detected by docTR. Returning best Tesseract result.")
            if best_result:
                best_result["ocr_strategy"] = "tesseract_multi_psm_no_blocks"
                return best_result
            return _make_result([_make_page_result("", 0.0)])

        # --- 2. Intelligent Masking (sidebar noise removal) ---
        h_img, w_img = image.shape[:2]
        sidebar_w = int(w_img * 0.15)
        sidebar_x = w_img - sidebar_w

        masked_image = image.copy()
        text_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        for block in blocks:
            for word_bbox in block.get("words", []):
                wx1, wy1, wx2, wy2 = word_bbox
                wx1, wy1 = max(0, wx1-5), max(0, wy1-10)
                wx2, wy2 = min(w_img, wx2+25), min(h_img, wy2+10)
                text_mask[wy1:wy2, wx1:wx2] = 1

        sidebar_area[sidebar_protection == 0] = 255
        masked_image[:, sidebar_x:] = sidebar_area

        # --- 3. Parallel Tesseract on blocks ---
        blocks.sort(key=lambda b: b["bbox"][1])

        def _process_block(args):
            """Process a single block crop with Tesseract."""
            idx, block = args
            x1, y1, x2, y2 = block["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            if x2 <= x1 or y2 <= y1:
                return idx, None

            crop = masked_image[y1:y2, x1:x2]

            # Heuristic: small blocks → metadata (चलानी, मिति)
            is_metadata = (y2 - y1) < 80 or (x2 - x1) < (w_img * 0.4)

            if is_metadata:
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                psm = "--psm 7"
            else:
                psm = "--psm 6"

            engine = TesseractOCREngine(
                lang=self.tesseract.lang,
                tesseract_config=psm,
                preprocess_config=self.preprocess_config
            )
            region_res = engine.process_image(crop)
            region_page = region_res["pages"][0]

            text = region_page["text"].strip()
            if not text:
                return idx, None

            # Adjust bounding boxes back to full-page coordinates
            for box in region_page["boxes"]:
                bx1, by1, bx2, by2 = box["bbox"]
                if is_metadata:
                    bx1, by1, bx2, by2 = bx1 // 2, by1 // 2, bx2 // 2, by2 // 2
                box["bbox"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]

            return idx, region_page

        # Run blocks in parallel (up to 6 threads)
        max_workers = min(len(blocks), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_process_block, enumerate(blocks)))

        # Sort by original index and stitch
        results.sort(key=lambda x: x[0])

        all_text_parts = []
        all_boxes = []
        all_confidences = []
        for _, page_data in results:
            if page_data is None:
                continue
            all_text_parts.append(page_data["text"])
            all_boxes.extend(page_data["boxes"])
            all_confidences.append(page_data["confidence"])

        full_text = "\n\n".join(all_text_parts)
        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        t_slow = time.time() - t_start
        logger.info("[Hybrid] Slow path done in %.2fs — confidence=%.4f", t_slow, avg_conf)

        result = _make_result([_make_page_result(full_text, avg_conf, all_boxes)])
        result["ocr_strategy"] = "doctr_layout_parallel_tesseract"
        return result

    def process_pdf(self, pdf_path: str, poppler_path: Optional[str] = None) -> dict:
        """Per-page hybrid: each page independently evaluated."""
        images = _convert_pdf_to_images(pdf_path, poppler_path)
        cfg = self.preprocess_config

        pages = []
        for idx, pil_img in enumerate(images):
            preprocessed = preprocess_pil_image(pil_img, cfg)
            
            result = self.process_image(preprocessed)
            
            p = result["pages"][0]
            pages.append(p)
            logger.info("Hybrid processed page %d/%d", idx + 1, len(images))

        return _make_result(pages)


# ===================================================================
# Shared utilities
# ===================================================================

def _convert_pdf_to_images(pdf_path: str, poppler_path: Optional[str] = None) -> list:
    """
    Convert all pages of a PDF into high-quality reference images.

    Args:
        pdf_path (str): File system path to the PDF.
        poppler_path (Optional[str]): System path to Poppler binaries.

    Returns:
        list[PIL.Image.Image]: List of converted images.

    Performance Note: 
    150 DPI is chosen as the "sweet spot" for Tesseract Devanagari OCR. 
    Higher DPI slows processing by 3x without a measurable accuracy gain.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise OCRError(
            "pdf2image is required for PDF processing. "
            "Install with: pip install pdf2image"
        ) from exc

    try:
        kwargs = {"pdf_path": pdf_path, "dpi": 150, "thread_count": 4}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        images = convert_from_path(**kwargs)
        logger.info("Converted PDF to %d page(s) at 150 DPI", len(images))
        return images
    except Exception as exc:
        raise OCRError(f"Failed to convert PDF to images: {exc}") from exc


# ===================================================================
# Backward-compatible OCREngine wrapper
# ===================================================================

class OCREngine:
    """
    High-level entry point for all documents (Images, PDF, Word).
    
    This is the primary class used by the `main.py` controller. It handles 
    pre-OCR tasks (like extension checking and born-digital text extraction) 
    and maintains backward compatibility for existing callers.
    """

    def __init__(
        self,
        lang: str = "nep+eng",
        tesseract_config: str = "--psm 3",
        poppler_path: Optional[str] = None,
        confidence_threshold: float = 0.85,
        preprocess_config: Optional[dict] = None,
    ):
        self.lang = lang
        self.tesseract_config = tesseract_config
        self.poppler_path = poppler_path
        self.preprocess_config = preprocess_config

        self._hybrid = HybridOCREngine(
            confidence_threshold=confidence_threshold,
            lang=lang,
            tesseract_config=tesseract_config,
            preprocess_config=preprocess_config,
        )

    # ------------------------------------------------------------------
    # Backward-compatible API  (returns list[str])
    # ------------------------------------------------------------------
    def process(self, file_path: str) -> list[str]:
        """
        Legacy text extraction API used for language detection and simple text views with via endpoint detect-language.

        Args:
            file_path (str): File system path to the document.

        Returns:
            list[str]: One string for each detected page.
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

        if ext in SUPPORTED_WORD_EXTENSIONS:
            logger.info("Processing Word document: %s", file_path)
            return self._process_word(str(path))

        if ext in SUPPORTED_PDF_EXTENSIONS:
            logger.info("Processing PDF: %s", file_path)
            # Try direct text extraction first (born-digital PDFs)
            direct_text = self._process_pdf_direct(str(path))
            if direct_text and any(page.strip() for page in direct_text):
                logger.info("Direct extraction successful for PDF")
                return direct_text
            logger.info("Direct extraction empty — falling back to OCR")
            result = self._hybrid.process_pdf(str(path), self.poppler_path)
            return [p["text"] for p in result["pages"]]

        # Image
        logger.info("Processing image: %s", file_path)
        preprocessed = preprocess_image(str(path), self.preprocess_config)
        result = self._hybrid.process_image(preprocessed)

        return [p["text"] for p in result["pages"]]

    # ------------------------------------------------------------------
    # Detailed API  (returns structured dict)
    # ------------------------------------------------------------------
    def process_detailed(self, file_path: str) -> dict:
        """
        Full feature extraction API for modern web-based result views.

        This is the method used by the `/upload` endpoint in `main.py`. It 
        returns the text contents alongside full bounding boxes and 
        confidence scores for every word detected(that's why it is called detailed).

        Args:
            file_path (str): File system path to the document.

        Returns:
            dict: Structured data containing pages, text, and bboxes.
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

        if ext in SUPPORTED_WORD_EXTENSIONS:
            texts = self._process_word(str(path))
            return _make_result(
                [_make_page_result(t, 1.0) for t in texts]
            )

        if ext in SUPPORTED_PDF_EXTENSIONS:
            direct_text = self._process_pdf_direct(str(path))
            if direct_text and any(page.strip() for page in direct_text):
                return _make_result(
                    [_make_page_result(t, 1.0) for t in direct_text]
                )
            
            result = self._hybrid.process_pdf(str(path), self.poppler_path)
            return result

        # Image
        preprocessed = preprocess_image(str(path), self.preprocess_config)
        result = self._hybrid.process_image(preprocessed)
        
        return result

    # ------------------------------------------------------------------
    # Word extraction
    # ------------------------------------------------------------------
    def _process_word(self, file_path: str) -> list[str]:
        """Extract text from a .doc or .docx file."""
        path = Path(file_path)
        ext = path.suffix.lower()

        try:
            if ext == ".docx":
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                return [text.strip()]
            elif ext == ".doc":
                import docx2txt
                text = docx2txt.process(file_path)
                return [text.strip()]
            else:
                raise OCRError(f"Unsupported word extension: {ext}")
        except Exception as e:
            logger.error(f"Word extraction failed: {e}")
            raise OCRError(f"Failed to extract text from Word document: {e}")

    # ------------------------------------------------------------------
    # Direct PDF text extraction (PyMuPDF)
    # ------------------------------------------------------------------
    def _process_pdf_direct(self, pdf_path: str) -> list[str]:
        """Extract text directly from a born-digital PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            page_texts = []
            for page in doc:
                text = page.get_text("text")
                page_texts.append(text.strip())
            return page_texts
        except Exception as e:
            logger.warning(f"Direct PDF extraction failed: {e}")
            return []



# (In OCREngine, the main entry point is via `process_detailed` for the /upload API,
#  while the simpler `process` is used for language detection via /detect-language)
#
# 1. Born-Digital/Word Check: 
#    If the file is a Born-Digital PDF or Word doc, it uses PyMuPDF or docx2txt directly.
#
# 2. Hybrid OCR Entry: 
#    If it's a scanned Image or PDF, `HybridOCREngine` takes over.
#
# 3. Fast Path (Tesseract):
#    It tries Tesseract with PSM 4, 6, and 3.
#    - Optimization: If it sees > 500 chars and > 85% confidence, it exits early to save time.
#    - Fallback: If ALL modes fail to extract at least 100 characters, it triggers the "Slow Path."
#
# 4. Slow Path (docTR + Parallel Tesseract):
#    - Uses `DocTROCREngine.get_blocks` to identify exact text regions (bounding boxes).
#    - Crops those regions and runs Tesseract (Multi-Threaded) on each crop individually.
#    - Merges the result for the final high-precision output.
#
# 5. PDF Workflow:
#    `HybridOCREngine.process_pdf` uses `_convert_pdf_to_images` (at 150 DPI) and 
#    loops over the `process_image` logic (Fast/Slow Path) for every page.
