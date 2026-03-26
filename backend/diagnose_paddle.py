import os
import sys
import logging
import cv2
import numpy as np
from PIL import Image
import fitz

# Ensure we are in the backend dir to find our modules
sys.path.append(os.getcwd())

from ocr.ocr_engine import _get_paddle_ocr
from ocr.preprocessing import preprocess_pil_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnose")

def diagnose_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 1. Test Sharpen=True (Current state)
    img_sharpen = preprocess_pil_image(pil_img, apply_threshold=False, sharpen=True)
    logger.info("--- Testing with SHARPEN=TRUE ---")
    ocr = _get_paddle_ocr()
    result = ocr.ocr(img_sharpen, cls=True)
    for res in result:
        for line in res:
            print(f"Conf: {line[1][1]:.3f} | Text: {line[1][0]}")

    # 2. Test Sharpen=False (Cleaner)
    img_clean = preprocess_pil_image(pil_img, apply_threshold=False, sharpen=False)
    logger.info("--- Testing with SHARPEN=FALSE ---")
    result = ocr.ocr(img_clean, cls=True)
    for res in result:
        for line in res:
            print(f"Conf: {line[1][1]:.3f} | Text: {line[1][0]}")

if __name__ == "__main__":
    test_pdf = "uploads/durga_laxmi_maharjan1.pdf"
    if os.path.exists(test_pdf):
        diagnose_ocr(test_pdf)
    else:
        print(f"File {test_pdf} not found")
