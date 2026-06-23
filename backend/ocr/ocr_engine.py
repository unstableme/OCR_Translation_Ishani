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
    preprocess_array,
    DEFAULT_CONFIG as PREPROCESS_DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_WORD_EXTENSIONS = {".docx", ".doc"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS | SUPPORTED_WORD_EXTENSIONS
PDF_DIRECT_TEXT_MIN_CHARS = 80
PDF_DIRECT_TEXT_MIN_WORDS = 12
PDF_SCANNER_ARTIFACT_LINES = {
    "camscanner",
    "scanned with camscanner",
    "scan with camscanner",
}
MIN_LAYOUT_REGION_CHARS = 80
FAST_PATH_PSM_MODES = ("--psm 3", "--psm 4", "--psm 6", "--psm 11", "--psm 1")


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


def _strip_pdf_text_artifacts(page_texts: list[str]) -> list[str]:
    """Remove scanner watermark lines from direct PDF text extraction."""
    cleaned_pages = []
    for page_text in page_texts:
        lines = []
        for line in page_text.splitlines():
            normalized = " ".join(line.strip().lower().split())
            if normalized in PDF_SCANNER_ARTIFACT_LINES:
                continue
            lines.append(line)
        cleaned_pages.append("\n".join(lines).strip())
    return cleaned_pages


def _direct_pdf_text_stats(page_texts: list[str]) -> tuple[int, int]:
    combined = "\n".join(page_texts).strip()
    return len("".join(combined.split())), len(combined.split())


def _is_meaningful_direct_pdf_text(page_texts: list[str]) -> bool:
    char_count, word_count = _direct_pdf_text_stats(page_texts)
    return (
        char_count >= PDF_DIRECT_TEXT_MIN_CHARS
        or word_count >= PDF_DIRECT_TEXT_MIN_WORDS
    )


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
    _available_languages: set[str] | None = None

    def __init__(
        self,
        lang: str = "nep+eng",
        tesseract_config: str = "--psm 3",
        preprocess_config: Optional[dict] = None,
    ):
        self.lang = lang
        self.tesseract_config = tesseract_config
        self.preprocess_config = preprocess_config

    @classmethod
    def _get_available_languages(cls) -> set[str]:
        if cls._available_languages is None:
            try:
                cls._available_languages = set(pt.get_languages(config=""))
            except pt.TesseractError as exc:
                raise OCRError(f"Could not inspect Tesseract languages: {exc}") from exc
        return cls._available_languages

    def _validate_language_packs(self) -> None:
        requested = {part for part in self.lang.split("+") if part}
        missing = sorted(requested - self._get_available_languages())
        if not missing:
            return

        available = ", ".join(sorted(self._get_available_languages())) or "none"
        raise OCRError(
            "Missing Tesseract language data for "
            f"{', '.join(missing)}. Available languages: {available}. "
            "Install the requested traineddata files before OCR; otherwise "
            "Devanagari text may be misread as English gibberish."
        )

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
            self._validate_language_packs()

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

    # ------------------------------------------------------------------
    # OpenCV layout helpers
    # ------------------------------------------------------------------
    def _detect_text_line_boxes(self, image: np.ndarray) -> list[list[int]]:
        """
        Detect line-like printed-text boxes without assuming a document format.

        The detector is used only for layout decisions and cropping. It favors
        conservative text-line candidates and rejects large photo/illustration
        regions, so clean newspaper, book, notice, and form layouts can all be
        handled by the same downstream OCR logic.
        """
        h_img, w_img = image.shape[:2]
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            11,
        )

        kernel_w = max(16, w_img // 90)
        kernel_h = max(1, h_img // 700)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        joined = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[list[int]] = []
        page_area = float(h_img * w_img)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < max(24, int(w_img * 0.018)):
                continue
            if h < 5 or h > max(120, int(h_img * 0.12)):
                continue
            if (w * h) > page_area * 0.22:
                continue

            crop = binary[y:y + h, x:x + w]
            ink_ratio = float(cv2.countNonZero(crop)) / float(max(1, w * h))
            if ink_ratio < 0.035 or ink_ratio > 0.72:
                continue

            boxes.append([x, y, x + w, y + h])

        merged = self._merge_nearby_boxes(boxes)
        return self._split_wide_line_boxes(merged, binary, image.shape)

    def _merge_nearby_boxes(self, boxes: list[list[int]]) -> list[list[int]]:
        """Merge line fragments split by punctuation, short words, or ligatures."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        heights = [b[3] - b[1] for b in boxes]
        median_h = float(np.median(heights)) if heights else 12.0
        y_tol = max(6, int(median_h * 0.65))
        x_gap_tol = max(10, int(median_h * 1.4))

        merged: list[list[int]] = []
        for box in boxes:
            if not merged:
                merged.append(box)
                continue

            prev = merged[-1]
            vertical_overlap = min(prev[3], box[3]) - max(prev[1], box[1])
            same_line = (
                vertical_overlap >= min(prev[3] - prev[1], box[3] - box[1]) * 0.45
                or abs(prev[1] - box[1]) <= y_tol
            )
            close_x = box[0] - prev[2] <= x_gap_tol

            if same_line and close_x:
                prev[0] = min(prev[0], box[0])
                prev[1] = min(prev[1], box[1])
                prev[2] = max(prev[2], box[2])
                prev[3] = max(prev[3], box[3])
            else:
                merged.append(box)

        return merged

    def _split_wide_line_boxes(
        self,
        boxes: list[list[int]],
        binary: np.ndarray,
        image_shape: tuple[int, int],
    ) -> list[list[int]]:
        """
        Split rows that were accidentally merged across multiple columns.

        Newspaper/article pages often have columns whose baselines align. The
        dilation step can connect those separate lines into one page-wide box,
        especially in photographed paper. Horizontal projection recovers the
        real text runs while leaving large headlines untouched.
        """
        if not boxes:
            return []

        h_img, w_img = image_shape[:2]
        split_boxes: list[list[int]] = []
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if width < w_img * 0.72 or height > max(75, int(h_img * 0.07)):
                split_boxes.append(box)
                continue

            crop = binary[y1:y2, x1:x2]
            projection = np.count_nonzero(crop, axis=0).astype(np.float32)
            smooth_width = max(17, int(w_img * 0.013))
            if smooth_width % 2 == 0:
                smooth_width += 1
            smoothed = cv2.blur(projection.reshape(1, -1), (smooth_width, 1)).ravel()
            threshold = max(1.0, float(np.percentile(smoothed, 70)) * 0.35)
            active = smoothed > threshold

            runs: list[list[int]] = []
            start = None
            for idx, is_active in enumerate(active):
                if is_active and start is None:
                    start = idx
                elif not is_active and start is not None:
                    runs.append([start, idx])
                    start = None
            if start is not None:
                runs.append([start, width])

            merged_runs: list[list[int]] = []
            max_word_gap = max(18, int(w_img * 0.02))
            for run in runs:
                if merged_runs and run[0] - merged_runs[-1][1] < max_word_gap:
                    merged_runs[-1][1] = run[1]
                else:
                    merged_runs.append(run)

            min_piece_w = max(55, int(w_img * 0.035))
            pieces = [
                [x1 + run[0], y1, x1 + run[1], y2]
                for run in merged_runs
                if (run[1] - run[0]) >= min_piece_w
            ]

            if len(pieces) >= 2:
                split_boxes.extend(pieces)
            else:
                split_boxes.append(box)

        return sorted(split_boxes, key=lambda b: (b[1], b[0]))

    def _group_lines_into_regions(
        self,
        line_boxes: list[list[int]],
        image_shape: tuple[int, int],
    ) -> list[dict]:
        """Group detected text lines into OCR regions such as headings or columns."""
        if not line_boxes:
            return []

        h_img, w_img = image_shape[:2]
        line_boxes = sorted(line_boxes, key=lambda b: (b[1], b[0]))
        median_h = float(np.median([b[3] - b[1] for b in line_boxes]))
        max_vertical_gap = max(18, int(median_h * 2.7))
        regions: list[dict] = []

        def horizontal_overlap(a: list[int], b: list[int]) -> float:
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            return overlap / float(max(1, min(a[2] - a[0], b[2] - b[0])))

        for line in line_boxes:
            best_idx = None
            best_score = 0.0
            for idx, region in enumerate(regions):
                bbox = region["bbox"]
                vertical_gap = line[1] - bbox[3]
                if vertical_gap < -median_h or vertical_gap > max_vertical_gap:
                    continue

                overlap = horizontal_overlap(bbox, line)
                region_width = bbox[2] - bbox[0]
                line_width = line[2] - line[0]
                width_ratio = min(region_width, line_width) / float(max(region_width, line_width))
                both_wide = region_width >= w_img * 0.62 and line_width >= w_img * 0.62
                comparable_width = width_ratio >= 0.42
                center_delta = abs(((line[0] + line[2]) / 2.0) - ((bbox[0] + bbox[2]) / 2.0))
                max_center_delta = max(region_width, line_width) * 0.28
                same_column = (
                    (overlap >= 0.35 and (comparable_width or both_wide))
                    or (center_delta <= max_center_delta and comparable_width)
                )

                if same_column:
                    score = overlap + (1.0 / (1.0 + max(0, vertical_gap)))
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            if best_idx is None:
                regions.append({"bbox": line.copy(), "lines": [line]})
                continue

            region = regions[best_idx]
            bbox = region["bbox"]
            bbox[0] = min(bbox[0], line[0])
            bbox[1] = min(bbox[1], line[1])
            bbox[2] = max(bbox[2], line[2])
            bbox[3] = max(bbox[3], line[3])
            region["lines"].append(line)

        filtered = []
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            width = x2 - x1
            height = y2 - y1
            if width < max(30, int(w_img * 0.025)) or height < 7:
                continue
            if width * height > h_img * w_img * 0.35:
                continue
            filtered.append(region)

        return self._sort_regions_for_reading(filtered, w_img)

    def _sort_regions_for_reading(self, regions: list[dict], page_width: int) -> list[dict]:
        """
        Sort OCR regions in a practical reading order.

        Full-width headings/captions keep top-to-bottom order. Narrow body
        regions are grouped into columns left-to-right, with top-to-bottom
        order inside each column. Single-column pages naturally collapse to
        one column.
        """
        if not regions:
            return []

        regions = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))
        wide_threshold = page_width * 0.68
        ordered: list[dict] = []
        pending: list[dict] = []

        def flush_pending():
            if not pending:
                return
            columns: list[list[dict]] = []
            for region in sorted(pending, key=lambda r: (r["bbox"][0], r["bbox"][1])):
                rx1, _, rx2, _ = region["bbox"]
                r_center = (rx1 + rx2) / 2.0
                placed = False
                for column in columns:
                    cx1 = min(r["bbox"][0] for r in column)
                    cx2 = max(r["bbox"][2] for r in column)
                    c_center = (cx1 + cx2) / 2.0
                    if abs(r_center - c_center) <= max((cx2 - cx1), (rx2 - rx1)) * 0.45:
                        column.append(region)
                        placed = True
                        break
                if not placed:
                    columns.append([region])

            columns.sort(key=lambda col: min(r["bbox"][0] for r in col))
            for column in columns:
                ordered.extend(sorted(column, key=lambda r: (r["bbox"][1], r["bbox"][0])))
            pending.clear()

        for region in regions:
            x1, _, x2, _ = region["bbox"]
            if (x2 - x1) >= wide_threshold:
                flush_pending()
                ordered.append(region)
            else:
                pending.append(region)
        flush_pending()
        return ordered

    def _detect_layout_regions(self, image: np.ndarray) -> list[dict]:
        line_boxes = self._detect_text_line_boxes(image)
        regions = self._group_lines_into_regions(line_boxes, image.shape)
        projection_regions = self._detect_projection_columns(image)
        if len(projection_regions) >= 2:
            regions = self._replace_overlapping_regions(regions, projection_regions)
        regions = self._remove_redundant_regions(regions, image.shape)
        return self._sort_regions_for_reading(regions, image.shape[1])

    def _detect_projection_columns(self, image: np.ndarray) -> list[dict]:
        """
        Detect multi-column body regions using vertical whitespace gutters.

        This supplements line detection for bulletin/newspaper pages where
        same-row columns can be visually close enough to touch after dilation.
        It scans the page for text bands first, then looks for side-by-side
        columns inside each band. That keeps the detector layout-agnostic: the
        columns may appear near the top, middle, bottom, or only in one section.
        """
        h_img, w_img = image.shape[:2]
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            11,
        )

        row_counts = np.count_nonzero(binary, axis=1)
        min_row_ink = max(3, int(w_img * 0.003))
        max_row_ink = int(w_img * 0.65)
        row_mask = ((row_counts >= min_row_ink) & (row_counts <= max_row_ink)).astype(np.uint8) * 255
        close_kernel_h = max(25, int(h_img * 0.035))
        row_mask = cv2.morphologyEx(
            row_mask.reshape(-1, 1),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_kernel_h)),
        ).ravel() > 0

        bands: list[list[int]] = []
        start = None
        for idx, is_active in enumerate(row_mask):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                bands.append([start, idx])
                start = None
        if start is not None:
            bands.append([start, h_img])

        min_band_h = max(55, int(h_img * 0.045))
        bands = [band for band in bands if (band[1] - band[0]) >= min_band_h]
        if not bands:
            return []

        regions = []
        for band_y1, band_y2 in bands:
            band = binary[band_y1:band_y2, :]
            band_row_counts = np.count_nonzero(band, axis=1)
            usable_rows = (band_row_counts >= min_row_ink) & (band_row_counts <= max_row_ink)
            if np.count_nonzero(usable_rows) < 8:
                continue

            projection = np.count_nonzero(band[usable_rows], axis=0).astype(np.float32)
            smooth_width = max(21, int(w_img * 0.018))
            if smooth_width % 2 == 0:
                smooth_width += 1
            smoothed = cv2.blur(projection.reshape(1, -1), (smooth_width, 1)).ravel()
            threshold = max(2.0, float(np.percentile(smoothed, 65)) * 0.35)
            active = smoothed > threshold

            runs: list[list[int]] = []
            run_start = None
            for idx, is_active in enumerate(active):
                if is_active and run_start is None:
                    run_start = idx
                elif not is_active and run_start is not None:
                    runs.append([run_start, idx])
                    run_start = None
            if run_start is not None:
                runs.append([run_start, w_img])

            merged_runs: list[list[int]] = []
            max_word_gap = max(18, int(w_img * 0.022))
            for run in runs:
                if merged_runs and run[0] - merged_runs[-1][1] < max_word_gap:
                    merged_runs[-1][1] = run[1]
                else:
                    merged_runs.append(run)

            min_column_w = max(85, int(w_img * 0.10))
            columns = [run for run in merged_runs if (run[1] - run[0]) >= min_column_w]
            if len(columns) < 2:
                continue

            for x1, x2 in columns:
                pad_x = max(4, int((x2 - x1) * 0.015))
                x1 = max(0, x1 - pad_x)
                x2 = min(w_img, x2 + pad_x)
                col = binary[band_y1:band_y2, x1:x2]
                col_row_counts = np.count_nonzero(col, axis=1)
                row_threshold = max(3, int((x2 - x1) * 0.018))
                rows = np.where(
                    (col_row_counts > row_threshold)
                    & (col_row_counts < (x2 - x1) * 0.60)
                )[0]
                if len(rows) < 5:
                    continue

                y1 = band_y1 + int(rows[0])
                y2 = band_y1 + int(rows[-1]) + 1
                if (y2 - y1) < max(45, int(h_img * 0.035)):
                    continue

                regions.append({
                    "bbox": [x1, y1, x2, y2],
                    "lines": [[x1, y1, x2, y2]],
                })

        return regions

    def _replace_overlapping_regions(
        self,
        regions: list[dict],
        replacements: list[dict],
    ) -> list[dict]:
        def overlap_fraction(a: list[int], b: list[int]) -> float:
            ix1 = max(a[0], b[0])
            iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2])
            iy2 = min(a[3], b[3])
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
            return intersection / float(area)

        kept = []
        for region in regions:
            bbox = region["bbox"]
            if any(overlap_fraction(bbox, replacement["bbox"]) > 0.35 for replacement in replacements):
                continue
            kept.append(region)
        kept.extend(replacements)
        return kept

    def _remove_redundant_regions(
        self,
        regions: list[dict],
        image_shape: tuple[int, int],
    ) -> list[dict]:
        """Drop tiny/one-line regions already covered by stronger text regions."""
        if len(regions) < 2:
            return regions

        h_img, w_img = image_shape[:2]

        def area(box: list[int]) -> int:
            return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

        def intersection_area(a: list[int], b: list[int]) -> int:
            ix1 = max(a[0], b[0])
            iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2])
            iy2 = min(a[3], b[3])
            return max(0, ix2 - ix1) * max(0, iy2 - iy1)

        kept = []
        for idx, region in enumerate(regions):
            bbox = region["bbox"]
            region_area = max(1, area(bbox))
            line_count = len(region.get("lines", []))
            region_width = bbox[2] - bbox[0]
            region_height = bbox[3] - bbox[1]

            if (
                line_count <= 1
                and region_width >= w_img * 0.60
                and region_height <= max(24, h_img * 0.022)
            ):
                continue

            if line_count <= 1 and region_width >= w_img * 0.55:
                nearby_columns = 0
                for other_idx, other in enumerate(regions):
                    if other_idx == idx or len(other.get("lines", [])) <= 3:
                        continue
                    other_bbox = other["bbox"]
                    x_overlap = max(0, min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0]))
                    y_gap = other_bbox[1] - bbox[3]
                    if x_overlap >= min(region_width, other_bbox[2] - other_bbox[0]) * 0.20 and -20 <= y_gap <= 80:
                        nearby_columns += 1
                if nearby_columns >= 2:
                    continue

            if line_count > 1:
                kept.append(region)
                continue

            covered_area = 0
            for other_idx, other in enumerate(regions):
                if other_idx == idx or len(other.get("lines", [])) <= 1:
                    continue
                other_bbox = other["bbox"]
                if area(other_bbox) <= region_area:
                    continue
                covered_area += intersection_area(bbox, other_bbox)

            if covered_area / float(region_area) < 0.55:
                kept.append(region)

        return kept

    def _is_complex_layout(self, regions: list[dict], image_width: int) -> bool:
        if len(regions) < 2:
            return False
        narrow = [r for r in regions if (r["bbox"][2] - r["bbox"][0]) < image_width * 0.68]
        if len(narrow) < 2:
            return False

        min_column_gap = image_width * 0.18
        for idx, left_region in enumerate(narrow):
            left = left_region["bbox"]
            left_center = (left[0] + left[2]) / 2.0
            left_h = max(1, left[3] - left[1])
            for right_region in narrow[idx + 1:]:
                right = right_region["bbox"]
                right_center = (right[0] + right[2]) / 2.0
                if abs(right_center - left_center) < min_column_gap:
                    continue

                overlap_y = max(0, min(left[3], right[3]) - max(left[1], right[1]))
                right_h = max(1, right[3] - right[1])
                vertical_overlap = overlap_y / float(min(left_h, right_h))
                if vertical_overlap >= 0.20 or len(narrow) >= 3:
                    return True

        return False

    def _process_regions_with_tesseract(
        self,
        image: np.ndarray,
        regions: list[dict],
        region_source: Optional[np.ndarray] = None,
    ) -> dict:
        source = region_source if region_source is not None else image
        h_img, w_img = source.shape[:2]

        def _process_region(args):
            idx, region = args
            x1, y1, x2, y2 = region["bbox"]
            pad_x = max(8, int((x2 - x1) * 0.025))
            pad_y = max(6, int((y2 - y1) * 0.08))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w_img, x2 + pad_x)
            y2 = min(h_img, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                return idx, None

            crop = source[y1:y2, x1:x2]
            if region_source is not None and len(crop.shape) == 3:
                crop = preprocess_array(crop, self.preprocess_config)
            line_count = len(region.get("lines", []))
            crop_height = y2 - y1
            if line_count <= 1 and crop_height < 80:
                psm = "--psm 7"
            else:
                psm = "--psm 6"

            if crop.shape[0] < 45:
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                scaled = True
            else:
                scaled = False

            engine = TesseractOCREngine(
                lang=self.tesseract.lang,
                tesseract_config=psm,
                preprocess_config=self.preprocess_config,
            )
            region_res = engine.process_image(crop)
            page = region_res["pages"][0]
            if not page["text"].strip():
                return idx, None

            for box in page["boxes"]:
                bx1, by1, bx2, by2 = box["bbox"]
                if scaled:
                    bx1, by1, bx2, by2 = bx1 // 2, by1 // 2, bx2 // 2, by2 // 2
                box["bbox"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]

            return idx, page

        max_workers = min(len(regions), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_process_region, enumerate(regions)))

        results.sort(key=lambda item: item[0])
        text_parts = []
        boxes = []
        confidences = []
        for _, page in results:
            if page is None:
                continue
            text_parts.append(page["text"].strip())
            boxes.extend(page["boxes"])
            confidences.append(page["confidence"])

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return _make_result([_make_page_result("\n\n".join(text_parts), avg_conf, boxes)])

    def process_image(self, image: np.ndarray, layout_image: Optional[np.ndarray] = None) -> dict:
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
        # --psm 3: Fully automatic segmentation (general multi-column fallback)
        # --psm 4: Single column of variable sizes (best for government letters)
        # --psm 6: Uniform block of text (good for body-heavy documents)
        # --psm 11: Sparse text in no particular order (screenshots/forms)
        # --psm 1: Automatic segmentation with orientation/script detection
        # ============================================================
        best_result = None
        best_text_len = 0
        best_conf = 0.0
        best_psm = ""

        for psm in FAST_PATH_PSM_MODES:
            engine = TesseractOCREngine(
                lang=self.tesseract.lang,
                tesseract_config=psm,
                preprocess_config=self.preprocess_config,
            )
            try:
                result = engine.process_image(image)
            except OCRError as exc:
                logger.warning("[Hybrid] %s failed: %s", psm, exc)
                continue

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

        t_fast = time.time() - t_start
        logger.info("[Hybrid] Fast path complete in %.2fs — best_psm=%s, best_text_len=%d, best_conf=%.4f",
                    t_fast, best_psm, best_text_len, best_conf)

        # Detect printed text regions even when the global pass returned text.
        # Multi-column pages can otherwise look "successful" while being
        # incomplete or out of reading order.
        layout_source = layout_image if layout_image is not None else image
        regions = self._detect_layout_regions(layout_source)
        complex_layout = self._is_complex_layout(regions, layout_source.shape[1])
        logger.info(
            "[Hybrid] OpenCV layout detected %d region(s), complex_layout=%s",
            len(regions), complex_layout,
        )

        if regions and (complex_layout or best_conf < self.threshold):
            layout_result = self._process_regions_with_tesseract(
                image,
                regions,
                region_source=layout_source if layout_image is not None else None,
            )
            layout_page = layout_result["pages"][0]
            layout_text_len = len(layout_page["text"].strip())
            layout_conf = layout_page["confidence"]
            logger.info(
                "[Hybrid] Layout path: text_len=%d, conf=%.4f",
                layout_text_len,
                layout_conf,
            )

            if (
                layout_text_len >= MIN_LAYOUT_REGION_CHARS
                and (
                    complex_layout
                    or layout_text_len >= int(best_text_len * 1.08)
                    or layout_conf >= best_conf
                )
            ):
                layout_result["ocr_strategy"] = "opencv_layout_tesseract"
                return layout_result

        # Accept fast path if Tesseract extracted meaningful content and the
        # page does not appear to need regional layout handling.
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

        sidebar_area = masked_image[:, sidebar_x:].copy()
        sidebar_protection = text_mask[:, sidebar_x:]
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
        strategies = []
        for idx, pil_img in enumerate(images):
            page_rgb = np.array(pil_img.convert("RGB"))
            page_bgr = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR)
            preprocessed = preprocess_pil_image(pil_img, cfg)
            
            result = self.process_image(preprocessed, layout_image=page_bgr)
            strategy = result.get("ocr_strategy")
            if strategy:
                strategies.append(strategy)
            
            p = result["pages"][0]
            pages.append(p)
            logger.info("Hybrid processed page %d/%d", idx + 1, len(images))

        pdf_result = _make_result(pages)
        if strategies:
            pdf_result["ocr_strategy"] = "+".join(dict.fromkeys(strategies))
        return pdf_result


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
            direct_text = _strip_pdf_text_artifacts(self._process_pdf_direct(str(path)))
            if direct_text and _is_meaningful_direct_pdf_text(direct_text):
                logger.info("Direct extraction successful for PDF")
                return direct_text
            char_count, word_count = _direct_pdf_text_stats(direct_text)
            logger.info(
                "Direct extraction not meaningful (chars=%d, words=%d); falling back to OCR",
                char_count,
                word_count,
            )
            result = self._hybrid.process_pdf(str(path), self.poppler_path)
            return [p["text"] for p in result["pages"]]

        # Image
        logger.info("Processing image: %s", file_path)
        original = cv2.imread(str(path))
        if original is None:
            raise OCRError(f"Could not read image from path: {file_path}")
        preprocessed = preprocess_array(original, self.preprocess_config)
        result = self._hybrid.process_image(preprocessed, layout_image=original)

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
            direct_text = _strip_pdf_text_artifacts(self._process_pdf_direct(str(path)))
            if direct_text and _is_meaningful_direct_pdf_text(direct_text):
                return _make_result(
                    [_make_page_result(t, 1.0) for t in direct_text]
                )

            char_count, word_count = _direct_pdf_text_stats(direct_text)
            logger.info(
                "Direct extraction not meaningful (chars=%d, words=%d); falling back to OCR",
                char_count,
                word_count,
            )
            result = self._hybrid.process_pdf(str(path), self.poppler_path)
            return result

        # Image
        original = cv2.imread(str(path))
        if original is None:
            raise OCRError(f"Could not read image from path: {file_path}")
        preprocessed = preprocess_array(original, self.preprocess_config)
        result = self._hybrid.process_image(preprocessed, layout_image=original)
        
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
