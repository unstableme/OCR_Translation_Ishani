"""
Image Preprocessing Module
===========================
Production-grade preprocessing pipeline for scanned document OCR.
Each step is independently configurable via a config dict.

Supports: deskew, denoise, contrast enhancement (CLAHE), Sauvola-style
binarization, morphological cleanup, line/border removal, and sharpening.
"""

import os
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration — every step can be toggled independently
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "resize": True,
    "max_dim": 2500,
    "deskew": True,
    "remove_colors": True,       # Mask out Blue stamps and Red signatures
    "denoise": True,
    "contrast_enhance": True,
    "binarize": False,            # Transition to Soft (Grayscale) OCR
    "morphological_cleanup": False,
    "remove_lines": False,         # IMPORTANT: Can destroy Devanagari Shirorekas
    "remove_vertical_noise": False, 
    "remove_sidebar_noise": False,  
    "thin_characters": False,       # Erosion can hurt LSTM accuracy
    "sharpen": False,
    "debug": False,
    "debug_output_dir": "debug_preprocessing",
}


def preprocess_image(image_path: str, config: Optional[dict] = None) -> np.ndarray:
    """
    Load an image from disk and run the full preprocessing pipeline.

    Returns
    -------
    numpy.ndarray
        Preprocessed grayscale image ready for OCR.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    logger.debug("Loaded image %s (%dx%d)", image_path, image.shape[1], image.shape[0])
    return preprocess_array(image, config)


def preprocess_pil_image(pil_image: Image.Image, config: Optional[dict] = None) -> np.ndarray:
    """
    Accept a PIL Image (e.g. from PDF conversion) and run the pipeline.
    """
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    logger.debug("Converted PIL image (%dx%d) for preprocessing", bgr.shape[1], bgr.shape[0])
    return preprocess_array(bgr, config)


def preprocess_array(image: np.ndarray, config: Optional[dict] = None) -> np.ndarray:
    """
    Core entry point: apply the full configurable preprocessing pipeline
    on a BGR numpy array.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    debug = cfg["debug"]
    debug_dir = cfg["debug_output_dir"]

    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        _save_debug(debug_dir, "00_original", image)

    # 1. Resize
    if cfg["resize"]:
        image = resize_image(image, max_dim=cfg["max_dim"])
        if debug:
            _save_debug(debug_dir, "01_resized", image)

    # 2. Remove Colored Artifacts (Stamps/Signatures) BEFORE grayscale
    if cfg["remove_colors"] and len(image.shape) == 3:
        image = remove_colored_artifacts(image)
        if debug:
            _save_debug(debug_dir, "01b_colors_removed", image)

    # 3. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # 4. Deskew
    if cfg["deskew"]:
        gray = deskew(gray)
        if debug:
            _save_debug(debug_dir, "02_deskewed", gray)

    # 4. Denoise
    if cfg["denoise"]:
        gray = denoise(gray)
        if debug:
            _save_debug(debug_dir, "03_denoised", gray)

    # 5. Contrast Enhancement (CLAHE)
    if cfg["contrast_enhance"]:
        gray = enhance_contrast(gray)
        if debug:
            _save_debug(debug_dir, "04_contrast", gray)

    # 6. Binarization (Sauvola-style)
    if cfg["binarize"]:
        gray = binarize_sauvola(gray)
        if debug:
            _save_debug(debug_dir, "05_binarized", gray)

    # 7. Morphological cleanup
    if cfg["morphological_cleanup"]:
        gray = morphological_cleanup(gray)
        if debug:
            _save_debug(debug_dir, "06_morphological", gray)

    # 8. Line / border removal
    if cfg["remove_lines"]:
        gray = remove_lines(gray)
        if debug:
            _save_debug(debug_dir, "07_lines_removed", gray)

    # 8b. Target vertical noise specifically
    if cfg["remove_vertical_noise"]:
        gray = remove_vertical_noise(gray)
        if debug:
            _save_debug(debug_dir, "07b_vertical_noise_removed", gray)

    # 8c. Aggressive Sidebar Cleaning (Right 15%)
    if cfg["remove_sidebar_noise"]:
        gray = remove_sidebar_noise(gray)
        if debug:
            _save_debug(debug_dir, "07c_sidebar_cleaned", gray)

    # 8d. Thin characters (Erosion)
    if cfg["thin_characters"]:
        gray = thin_characters(gray)
        if debug:
            _save_debug(debug_dir, "07d_thinned", gray)

    # 9. Sharpening (optional)
    if cfg["sharpen"]:
        gray = sharpen(gray)
        if debug:
            _save_debug(debug_dir, "08_sharpened", gray)

    logger.debug("Preprocessing complete — output shape %s", gray.shape)
    return gray


# ===================================================================
# Individual pipeline steps
# ===================================================================

def resize_image(image: np.ndarray, max_dim: int = 2500) -> np.ndarray:
    """Resize image if longest side exceeds *max_dim*, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect skew angle via Hough Line Transform and rotate to correct.
    Falls back to no-op if no reliable angle is found.
    """
    # Edge detection for line finding
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough — gives line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=gray.shape[1] // 4,
                            maxLineGap=20)
    if lines is None:
        logger.debug("Deskew: no lines found, skipping")
        return gray

    # Collect angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (±30°)
        if abs(angle) < 30:
            angles.append(angle)

    if not angles:
        logger.debug("Deskew: no near-horizontal lines, skipping")
        return gray

    median_angle = float(np.median(angles))

    # Don't bother rotating for very small angles
    if abs(median_angle) < 0.3:
        logger.debug("Deskew: angle %.2f° too small, skipping", median_angle)
        return gray

    logger.info("Deskew: correcting %.2f° skew", median_angle)
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def denoise(gray: np.ndarray) -> np.ndarray:
    """
    Non-local Means denoising.  Conservative strength (h=10) to
    preserve text edges — critical for Devanagari script.
    """
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                     searchWindowSize=21)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation (CLAHE).
    Boosted for soft grayscale OCR input.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Further normalize to stretch the full 0-255 range
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)


def binarize_sauvola(gray: np.ndarray, window_size: int = 25,
                      k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """
    Sauvola-inspired binarization: threshold = mean * (1 + k * (std/R - 1)).
    Robust to uneven lighting — better than simple adaptive Gaussian.
    """
    # Integral images for fast local mean / std computation
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (window_size, window_size))
    mean_sq = cv2.blur(gray_f ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    threshold = mean * (1.0 + k * (std / R - 1.0))
    binary = np.where(gray_f > threshold, 255, 0).astype(np.uint8)
    return binary


def morphological_cleanup(gray: np.ndarray) -> np.ndarray:
    """
    Opening (remove small noise) → closing (strengthen text strokes).
    Kernel sizes tuned for typical 150–300 DPI scanned documents.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Opening: erode then dilate — removes small dots / specks
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing: dilate then erode — fills small gaps in text strokes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def remove_lines(gray: np.ndarray) -> np.ndarray:
    """Detect and remove horizontal / vertical lines using morphological ops."""
    h, w = gray.shape[:2]
    # Invert for morphology (text = white, background = black)
    inverted = cv2.bitwise_not(gray)

    # --- Horizontal lines ---
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
    horiz_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    # --- Vertical lines (standard) ---
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 15))
    vert_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vert_kernel, iterations=2)

    all_lines = cv2.add(horiz_lines, vert_lines)
    all_lines = cv2.dilate(all_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    result = gray.copy()
    result[all_lines > 0] = 255
    return result


def remove_vertical_noise(gray: np.ndarray) -> np.ndarray:
    """Specific morphological suppression for vertical scan noise/artifacts."""
    h, w = gray.shape[:2]
    inverted = cv2.bitwise_not(gray)
    # Aggressive vertical kernel for scan noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vert_noise = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=2)
    # Subtraction ensures we remove the noise while leaving text alone
    result = gray.copy()
    result[vert_noise > 0] = 255
    return result


def remove_colored_artifacts(image: np.ndarray) -> np.ndarray:
    """
    Use HSV masking to remove Blue stamps and Red signatures.
    Replaces matched regions with White (255, 255, 255).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Blue stamp ranges (adjust if stamps are too light/dark)
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (140, 255, 255))

    # Red signature ranges (wraps around 0/180)
    # MODIFIED: Red is often used for legitimate text (Chalani/Titles).
    # We will ONLY remove very vibrant red (typical of ink signatures)
    # and leave more "printed-looking" red alone.
    # Actually, the user says "we cant remove the red text including chalani and even title"
    # So we will DISABLE red removal for now to ensure no data loss.
    # red_mask1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
    # red_mask2 = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
    # red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_mask = np.zeros_like(blue_mask) # Disable red removal for accuracy

    # Combine masks
    mask = cv2.bitwise_or(blue_mask, red_mask)

    # Dilate mask slightly to prevent colored "halos"
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    result = image.copy()
    result[mask > 0] = [255, 255, 255]
    return result


def remove_sidebar_noise(gray: np.ndarray) -> np.ndarray:
    """Aggressively target and remove noise on the right sidebar (common in scanners)."""
    h, w = gray.shape[:2]
    # Define vertical "danger zone" (Narrowed to 1% for ABSOLUTE text safety)
    sidebar_w = int(w * 0.01)
    sidebar_x = w - sidebar_w
    
    sidebar = gray[:, sidebar_x:]
    inverted = cv2.bitwise_not(sidebar)
    
    # Target vertical stripes with a very high kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    stripes = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=3)
    
    # White out the detected stripes in the original sidebar
    cleaned_sidebar = sidebar.copy()
    cleaned_sidebar[stripes > 0] = 255
    
    result = gray.copy()
    result[:, sidebar_x:] = cleaned_sidebar
    return result


def thin_characters(gray: np.ndarray) -> np.ndarray:
    """Reduce thickness of characters (1x1 erosion) to prevent bloating."""
    kernel = np.ones((2, 2), np.uint8)
    # Note: erosion on white background image (255=bg, 0=text) 
    # needs to be Dilation on the black text pixels
    inverted = cv2.bitwise_not(gray)
    thinned = cv2.erode(inverted, kernel, iterations=1)
    return cv2.bitwise_not(thinned)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Unsharp mask sharpening for faint or slightly blurred text.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened


# ===================================================================
# Debug helper
# ===================================================================

def _save_debug(output_dir: str, step_name: str, image: np.ndarray):
    """Save an intermediate image for visual inspection."""
    path = os.path.join(output_dir, f"{step_name}.png")
    cv2.imwrite(path, image)
    logger.debug("Debug: saved %s", path)