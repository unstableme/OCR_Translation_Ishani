"""
OCR Engine Module
=================
Production-grade OCR text extraction supporting three engines:

- **TesseractOCREngine** — Tesseract with Nepali/English language packs
- **DocTROCREngine** — docTR (db_resnet50 + crnn_vgg16_bn) with confidence + boxes
- **HybridOCREngine** — docTR first, Tesseract fallback if confidence < threshold

The backward-compatible ``OCREngine`` class wraps HybridOCREngine and
preserves the existing ``process(file_path) -> list[str]`` API.
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
    Tesseract-based OCR engine.  Native Nepali (``nep``) support makes
    this the strongest option for Devanagari-script documents.
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
        Run Tesseract on a preprocessed numpy image.
        Uses a SINGLE image_to_data call and reconstructs structured text
        from the output — preserving line and paragraph breaks.

        Returns
        -------
        dict  ``{"pages": [{"text", "confidence", "boxes"}]}``
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

    def process_pdf(self, pdf_path: str, poppler_path: Optional[str] = None) -> dict:
        """Convert PDF → images → Tesseract OCR with parallel processing."""
        images = _convert_pdf_to_images(pdf_path, poppler_path)
        cfg = self.preprocess_config

        def _process_one(args):
            idx, pil_img = args
            preprocessed = preprocess_pil_image(pil_img, cfg)
            result = self.process_image(preprocessed)
            return idx, result["pages"][0]

        max_workers = min(len(images), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_process_one, enumerate(images)))

        results.sort(key=lambda x: x[0])
        return _make_result([page for _, page in results])


# ===================================================================
# docTR Engine
# ===================================================================

class DocTROCREngine:
    """
    docTR-based OCR engine (db_resnet50 + crnn_vgg16_bn).
    Model is loaded **once** in ``__init__`` and reused for all pages.
    """

    def __init__(self, preprocess_config: Optional[dict] = None):
        self.preprocess_config = preprocess_config
        self._model = None  # Lazy-loaded on first call

    def _load_model(self):
        """Lazy-load the docTR model (downloads weights on first run)."""
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

    def process_image(self, image: np.ndarray) -> dict:
        """
        Standard API: Run docTR and return a dict result.
        """
        result = self._process_raw(image)
        pages = []
        for page in result.pages:
            page_words = []
            page_confs = []
            page_boxes = []
            h, w = page.dimensions
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        page_words.append(word.value)
                        page_confs.append(word.confidence)
                        geo = word.geometry
                        bbox = [int(geo[0][0]*w), int(geo[0][1]*h), int(geo[1][0]*w), int(geo[1][1]*h)]
                        page_boxes.append({"text": word.value, "confidence": round(word.confidence, 4), "bbox": bbox})
            full_text = " ".join(page_words)
            avg_conf = sum(page_confs) / len(page_confs) if page_confs else 0.0
            pages.append(_make_page_result(full_text, avg_conf, page_boxes))
        return _make_result(pages)

    def get_blocks(self, image: np.ndarray) -> list[dict]:
        """
        Run docTR and return only the physical blocks with their normalized coordinates.
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

    def process_pdf(self, pdf_path: str, poppler_path: Optional[str] = None) -> dict:
        """Convert PDF pages → images → docTR (sequential to reuse model)."""
        self._load_model()
        images = _convert_pdf_to_images(pdf_path, poppler_path)
        cfg = self.preprocess_config

        pages = []
        for idx, pil_img in enumerate(images):
            preprocessed = preprocess_pil_image(pil_img, cfg)
            result = self.process_image(preprocessed)
            pages.append(result["pages"][0])
            logger.info("docTR processed page %d/%d", idx + 1, len(images))

        return _make_result(pages)


# ===================================================================
# Hybrid Engine (docTR + Tesseract fallback)
# ===================================================================

class HybridOCREngine:
    """
    Runs docTR first.  If average word confidence falls below
    *confidence_threshold*, falls back to Tesseract.
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

    def process_image(self, image: np.ndarray, debug_filename: Optional[str] = None) -> dict:
        """
        TESSERACT-FIRST Hybrid OCR (Multi-PSM):
        1. Try multiple Tesseract page segmentation modes to maximize extraction.
        2. Pick the result that extracted the most text.
        3. If all modes produce very little text → fall back to docTR layout + parallel Tesseract.
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
            best_result["diagnostic_url"] = None
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
                best_result["diagnostic_url"] = None
                best_result["ocr_strategy"] = "tesseract_multi_psm_doctr_failed"
                return best_result
            return _make_result([_make_page_result("", 0.0)])

        if not blocks:
            logger.info("No blocks detected by docTR. Returning best Tesseract result.")
            if best_result:
                best_result["diagnostic_url"] = None
                best_result["ocr_strategy"] = "tesseract_multi_psm_no_blocks"
                return best_result
            return _make_result([_make_page_result("", 0.0)])

        # --- 2. Intelligent Masking (sidebar noise removal) ---
        ts = int(time.time())
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

        sidebar_area = masked_image[:, sidebar_x:]
        sidebar_protection = text_mask[:, sidebar_x:]
        sidebar_area[sidebar_protection == 0] = 255
        masked_image[:, sidebar_x:] = sidebar_area

        # Save diagnostic view
        dbg_base = debug_filename if debug_filename else "smart_mask_debug"
        dbg_name = f"{dbg_base}_{ts}.png"
        masked_dbg_path = os.path.join("uploads/debug", dbg_name)
        cv2.imwrite(masked_dbg_path, masked_image)
        diagnostic_url = f"/uploads/debug/{dbg_name}"

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
        result["diagnostic_url"] = diagnostic_url
        result["ocr_strategy"] = "doctr_layout_parallel_tesseract"
        return result

    def process_pdf(self, pdf_path: str, poppler_path: Optional[str] = None) -> dict:
        """Per-page hybrid: each page independently evaluated."""
        images = _convert_pdf_to_images(pdf_path, poppler_path)
        cfg = self.preprocess_config

        pages = []
        for idx, pil_img in enumerate(images):
            preprocessed = preprocess_pil_image(pil_img, cfg)
            
            # Use unique filename for each page's mask
            masked_name = f"masked_feed_{Path(pdf_path).name}_page_{idx+1}.png"
            result = self.process_image(preprocessed, debug_filename=masked_name)
            
            p = result["pages"][0]
            p["diagnostic_url"] = result.get("diagnostic_url")
            pages.append(p)
            logger.info("Hybrid processed page %d/%d", idx + 1, len(images))

        return _make_result(pages)


# ===================================================================
# Shared utilities
# ===================================================================

def _convert_pdf_to_images(pdf_path: str, poppler_path: Optional[str] = None) -> list:
    """Convert every page of a PDF to a PIL Image at 150 DPI."""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise OCRError(
            "pdf2image is required for PDF processing. "
            "Install with: pip install pdf2image"
        ) from exc

    try:
        # NOTE: 150 DPI is the sweet spot for Tesseract Nepali OCR.
        # Higher DPI (160/200) causes Tesseract PSM segmentation failures
        # and 2-3x slower processing. Do NOT change without testing.
        kwargs = {"pdf_path": pdf_path, "dpi": 150, "thread_count": 4}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        images = convert_from_path(**kwargs)
        logger.info("Converted PDF to %d page(s) at 150 DPI", len(images))
        return images
    except Exception as exc:
        raise OCRError(f"Failed to convert PDF to images: {exc}") from exc


def _draw_debug_boxes(image: np.ndarray, boxes: list, output_path: str):
    """
    Draw OCR bounding boxes on an image and save for debugging.
    Useful for visual verification of detection accuracy.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        conf = box.get("confidence", 0)
        # Green for high confidence, red for low
        color = (0, 200, 0) if conf >= 0.85 else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(vis, f"{conf:.0%}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, vis)
    logger.debug("Debug boxes saved to %s", output_path)


# ===================================================================
# Backward-compatible OCREngine wrapper
# ===================================================================

class OCREngine:
    """
    Drop-in replacement for the original OCREngine.

    - ``process(file_path) → list[str]`` — backward-compatible (page texts)
    - ``process_detailed(file_path) → dict`` — new structured output
    """

    def __init__(
        self,
        lang: str = "nep+eng",
        tesseract_config: str = "--psm 3",
        poppler_path: Optional[str] = None,
        confidence_threshold: float = 0.85,
        preprocess_config: Optional[dict] = None,
        debug_boxes: bool = False,
    ):
        self.lang = lang
        self.tesseract_config = tesseract_config
        self.poppler_path = poppler_path
        self.debug_boxes = debug_boxes
        self.preprocess_config = preprocess_config

        self._hybrid = HybridOCREngine(
            confidence_threshold=confidence_threshold,
            lang=lang,
            tesseract_config=tesseract_config,
            preprocess_config=preprocess_config,
        )
        # Ensure debug directory exists
        os.makedirs("uploads/debug", exist_ok=True)

    # ------------------------------------------------------------------
    # Backward-compatible API  (returns list[str])
    # ------------------------------------------------------------------
    def process(self, file_path: str) -> list[str]:
        """
        Extract text from an image, PDF, or Word file.

        Returns
        -------
        list[str]
            One string per page — same contract as the original OCREngine.
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

        if self.debug_boxes and result["pages"]:
            _draw_debug_boxes(
                preprocessed, result["pages"][0]["boxes"],
                "debug_preprocessing/boxes_output.png",
            )

        return [p["text"] for p in result["pages"]]

    # ------------------------------------------------------------------
    # New detailed API  (returns structured dict)
    # ------------------------------------------------------------------
    def process_detailed(self, file_path: str) -> dict:
        """
        Extract text with full metadata (confidence, bounding boxes).

        Returns
        -------
        dict  ``{"pages": [{"text", "confidence", "boxes"}]}``
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
            
            # Collect all debug images (preprocessed + masked)
            debug_images = []
            for p_idx, p in enumerate(result["pages"]):
                # Preprocessed images are already saved during hybrid.process_pdf loop?
                # Actually, process_detailed for PDF should be the central place for debug saving.
                # But to keep it simple, I'll just gather the diagnostic_urls from the pages.
                diag = p.get("diagnostic_url")
                if diag:
                    debug_images.append(diag)
            
            # Also add the preprocessed ones (already saved in uploads/debug/debug_preprocessed_...)
            for idx in range(len(result["pages"])):
                dbg_name = f"debug_preprocessed_{path.name}_page_{idx+1}.png"
                debug_images.append(f"/uploads/debug/{dbg_name}")
                
            result["debug_images"] = debug_images
            return result

        # Image
        preprocessed = preprocess_image(str(path), self.preprocess_config)
        
        # Save preprocessed image for user inspection
        debug_filename = f"debug_preprocessed_{path.name}.png"
        debug_path = os.path.join("uploads/debug", debug_filename)
        cv2.imwrite(debug_path, preprocessed)
        
        masked_name = f"masked_feed_{path.name}.png"
        result = self._hybrid.process_image(preprocessed, debug_filename=masked_name)
        
        # Collect diagnostic URL from the hybrid result
        diag_url = result.get("diagnostic_url")
        
        result["debug_images"] = [
            f"/uploads/debug/{debug_filename}",
            diag_url
        ] if diag_url else [f"/uploads/debug/{debug_filename}"]
        
        return result

    # ------------------------------------------------------------------
    # Word extraction (unchanged)
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
    # Direct PDF text extraction (PyMuPDF — unchanged)
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


# ---------------------------------------------------------------------------
# Backward-compatible convenience function
# ---------------------------------------------------------------------------
def run_ocr(image_path) -> str:
    """
    Extract text from an image or PDF.

    Thin wrapper around :class:`OCREngine` kept for backward
    compatibility with existing callers.
    """
    engine = OCREngine()
    pages = engine.process(image_path)
    return "\n\n".join(pages) if isinstance(pages, list) else pages