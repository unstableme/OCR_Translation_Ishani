# NepText: Intelligent Himalayan OCR & Translation Suite

**NepText** is a premium AI-powered platform designed for high-accuracy text extraction and translation of Devanagari-based documents. It specialized in Himalayan languages including **Tamang**, **Newari (Nepal Bhasa)**, and **Nepali**.

---

## 🌟 Key Features

- **Hybrid OCR Pipeline**: High-performance text extraction using a combination of Tesseract and `docTR` for multi-stage processing.
- **Contextual Himalayan Translation**: Advanced LLM post-processing for culturally accurate translations from Tamang and Nepal Bhasa to Nepali.
- **Smart Camera Support**: 
  - **Mobile**: Integrated with native OS document scanners (Auto-crop, deskew).
  - **Desktop**: Custom in-browser webcam viewfinder with real-time scanning guides.
- **Multi-Format Support**: Processes `.jpg`, `.png`, `.pdf` (multi-page), and `.docx`.
- **Premium Dashboard**: Glassmorphism-inspired UI with smooth micro-animations and "Typewriter" effect results.
- **PDF Export**: One-click generation of processed results into professional PDF documents.

---

## 🏗️ Technical Stack

- **Frontend**: React 18, Vite, Lucide Icons, Vanilla CSS (Premium Glassmorphism).
- **Backend**: Python 3.10+, FastAPI, docTR (Deep Learning OCR), Tesseract.
- **AI Engine**: LLM-based translation optimization (OpenRouter/Gemini).
- **Deployment**: Docker & Docker Compose ready.

---

## 🚀 Deployment & Setup

### 1. Configure Environment
1. Clone the repository.
2. Rename `.env.example` to `.env` in the root and backend directories.
3. Add your `OPENROUTER_KEY` or relevant AI credentials.

### 2. Run with Docker (Recommended)
```bash
docker-compose up --build -d
```
Access via:
- **Frontend**: http://localhost
- **Backend Docs**: http://localhost:8000/docs

---

## 🛠️ Local Development & Mobile Testing

To test the **"Use Camera"** features on actual mobile devices, you must bind the servers to your network IP.

### 1. Start Backend
```bash
cd backend
# Recommended: Ensure local firewall allows port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Frontend
```bash
cd frontend
# Expose to network for mobile access
npm run dev -- --host
```

### 3. Mobile Access
Find your laptop's **IPv4 Address** (e.g., `192.168.1.15`) via `ipconfig` and open this on your phone:
`http://192.168.1.15:5173`

---

## 📂 Project Structure

```text
├── backend/            # FastAPI, OCR processing, LLM logic
├── frontend/           # React dashboard, Camera API, UI logic
├── docker-compose.yml  # Container orchestration
└── README.md           # Documentation
```

## 📜 License
*Private Project - All rights reserved.*
