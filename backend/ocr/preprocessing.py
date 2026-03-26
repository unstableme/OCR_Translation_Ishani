"""
Image Preprocessing Module
===========================
Prepares images for OCR by applying grayscale conversion,
Gaussian blur, and adaptive thresholding.
"""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str, apply_threshold: bool = True, sharpen: bool = False) -> np.ndarray:
    """
    Load an image from disk and apply the preprocessing pipeline.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    apply_threshold : bool, default True
        Whether to apply adaptive thresholding.
    sharpen : bool, default False
        Whether to apply a sharpening filter. Helps with blurry Devanagari.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    logger.debug("Loaded image %s (%dx%d)", image_path, image.shape[1], image.shape[0])
    return _apply_pipeline(image, apply_threshold=apply_threshold, sharpen=sharpen)


def preprocess_pil_image(pil_image: Image.Image, apply_threshold: bool = True, sharpen: bool = False) -> np.ndarray:
    """
    Accept a PIL Image (e.g. from PDF conversion) and apply the
    same preprocessing pipeline.
    """
    # Convert PIL → OpenCV BGR format
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    logger.debug("Converted PIL image (%dx%d) for preprocessing", bgr.shape[1], bgr.shape[0])
    return _apply_pipeline(bgr, apply_threshold=apply_threshold, sharpen=sharpen)


def _apply_pipeline(image: np.ndarray, apply_threshold: bool = True, sharpen: bool = False) -> np.ndarray:
    """
    Core preprocessing: resize → grayscale → sharpen → blur → adaptive threshold.
    """
    # 1. Resize if too large
    image = resize_image(image, max_dim=2500)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Optional Sharpening
    if sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)

    # 4. No blur (1,1 is effectively none) prevents softening complex scripts
    blur = cv2.GaussianBlur(gray, (1, 1), 0)

    # 5. Adaptive thresholding for contrast enhancement
    if not apply_threshold:
        return blur

    thresholded = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return thresholded


def resize_image(image: np.ndarray, max_dim: int = 2500) -> np.ndarray:
    """
    Resize image if its longest side exceeds max_dim, maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)