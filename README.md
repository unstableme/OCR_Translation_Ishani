# OCR & Translation Project

A full-stack application that extracts text from images/PDFs using OCR and translates it into Nepali. Built with **FastAPI** (Backend) and **React** (Frontend).

## Project Structure

- **`backend/`**: Python FastAPI server. Handles file uploads, OCR processing, and translation.
    - Uses `Tesseract` (via `pytesseract`) for OCR.
    - Uses `SQLAlchemy` for database management.
- **`frontend/`**: React application (Vite). Provides a user interface for uploading files and viewing results.

## Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **Tesseract OCR** installed on your system.

## Setup & Running

### 1. Backend Setup

Navigate to the `backend` directory:
```bash
cd backend
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the server:
```bash
uvicorn main:app --reload
```
The backend will run at `http://localhost:8000`.

### 2. Frontend Setup

Open a new terminal and navigate to the `frontend` directory:
```bash
cd frontend
```

Install dependencies:
```bash
npm install
```

Run the development server:
```bash
npm run dev
```
The frontend will run at `http://localhost:5173`.

## Usage

1. Open `http://localhost:5173`.
2. Login with any credentials (Mock Login: purely for UI demo).
3. Drag & drop an image or PDF.
4. Click **Translate Document**.
5. View the original extracted text and the translated Nepali text.
6. Download the result as a PDF.
