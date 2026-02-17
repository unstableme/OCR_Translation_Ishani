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


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and apply the preprocessing pipeline.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Preprocessed (thresholded) grayscale image.

    Raises
    ------
    ValueError
        If the image cannot be read from the given path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    logger.debug("Loaded image %s (%dx%d)", image_path, image.shape[1], image.shape[0])
    return _apply_pipeline(image)


def preprocess_pil_image(pil_image: Image.Image) -> np.ndarray:
    """
    Accept a PIL Image (e.g. from PDF conversion) and apply the
    same preprocessing pipeline.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Input image.

    Returns
    -------
    numpy.ndarray
        Preprocessed (thresholded) grayscale image.
    """
    # Convert PIL → OpenCV BGR format
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    logger.debug("Converted PIL image (%dx%d) for preprocessing", bgr.shape[1], bgr.shape[0])
    return _apply_pipeline(bgr)


def _apply_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Core preprocessing: resize → grayscale → blur → adaptive threshold.
    """
    # 1. Resize if too large
    image = resize_image(image, max_dim=2500)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. No blur (1,1 is effectively none) prevents softening complex scripts
    blur = cv2.GaussianBlur(gray, (1, 1), 0)

    # 4. Adaptive thresholding for contrast enhancement
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