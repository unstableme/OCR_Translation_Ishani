# NepText: Professional OCR & Translation Suite

**NepText** is a high-performance system designed for extracting and translating text from documents (images and PDFs). It specializes in Himalayan languages, providing seamless conversion to Nepali using state-of-the-art AI.

---

## 🌟 Key Features

- **Hybrid OCR Engine**: Combines Tesseract and `docTR` for high-accuracy text extraction across various document qualities.
- **Context-Aware Translation**: Uses advanced LLM post-processing to ensure culturally accurate and grammatically correct translations.
- **Language Detection**: Automatically identifies source languages (Nepali, Tamang, Newari, etc.) in scanned documents.
- **Multi-Format Support**: Handles `.jpg`, `.png`, `.pdf`, and `.docx` files.
- **Scalable Architecture**: Built with FastAPI for speed and Docker for seamless deployment.

---

## 🏗️ Technical Stack

- **Backend**: Python 3.10+, FastAPI, SQLAlchemy, PostgreSQL.
- **Frontend**: React (Vite), TailwindCSS, Modern UI.
- **OCR Tools**: `pytesseract`, `python-doctr`, `OpenCV`.
- **AI Processing**: LLM-based translation optimization.

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- An AI API Key (e.g., OpenRouter)

### Deployment with Docker (Recommended)

1. **Configure Environment**:
   - Rename `.env.example` to `.env` in the root directory.
   - Add your `OPENROUTER_KEY`.

2. **Run the Stack**:
   ```bash
   docker-compose up --build -d
   ```

3. **Access**:
   - **Frontend**: http://localhost
   - **API Docs**: http://localhost:8000/docs

---

## 🛠️ Local Development (Manual)

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# Activate venv (Source or .\venv\Scripts\activate)
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 📂 Project Structure

```text
├── backend/            # FastAPI server, OCR logic, and DB models
├── frontend/           # React application and UI components
├── docker-compose.yml  # Orchestration for containerized deployment
└── .github/workflows/  # CI/CD pipelines
```

## 📜 License
*Private Project - All rights reserved.*
