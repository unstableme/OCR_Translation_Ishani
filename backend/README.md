# NepText Backend: Himalayan OCR & AI Engine

The server-side component of the **NepText** platform, built for robust, deep-learning-driven OCR and context-aware Himalayan language translation.

---

## 🌟 Key Features

- **Hybrid OCR Engine**: High-accuracy text extraction combining Tesseract and `docTR` for multi-stage processing.
- **AI Translation Post-Processing**: Direct optimization using advanced LLMs (via OpenRouter/Gemini) specialized in Himalayan languages:
  - **Tamang** to **Nepali**.
  - **Newari (Nepal Bhasa)** to **Nepali**.
- **Multi-Format Ingestion**: Supports `.jpg`, `.png`, `.pdf` (multi-page), and `.docx`.
- **FastAPI Core**: Asynchronous backend for high-speed concurrent uploads.
- **Swagger Documentation**: Self-documenting API endpoints.

---

## 🏗️ Technical Stack

- **Framework**: `Python 3.10+`, `FastAPI`.
- **OCR Libraries**: `python-doctr` (PyTorch backend), `pytesseract`, `OpenCV`.
- **Database**: `SQLAlchemy` with `sqlite/PostgreSQL` support.
- **Environment**: `Docker` and `Docker Compose` ready.

---

## 🚀 Setup & Local Access

1. **Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API Server**:
   ```bash
   uvicorn main:app --reload
   ```

4. **Mobile/External Access**:
   To allow other devices (like your phone) on the same Wi-Fi to reach the API, bind it to all interfaces:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 🛠️ API & Endpoints

- **`/upload`**: Receives document files, extracts text, and translates it.
- **`/translate`**: Processes direct text input.
- **`/docs`**: Interactive Swagger documentation.

---

## 📜 Project Overview
This backend performs the heavy lifting for **NepText**, utilizing deep learning models to overcome Devanagari OCR challenges.

---
*NepText Backend Core - 2026*
