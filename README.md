# NepText

A professional AI-powered tool for OCR extraction and translation. **NepText** seamlessly extracts text from images and PDF documents and translates them into Nepali. Built with **FastAPI** (Backend) and **React** (Frontend).

## Project Structure

- **`backend/`**: Python FastAPI server. Handles file uploads, OCR processing, and translation.
    - Uses `Tesseract` for OCR and `OpenAI/OpenRouter` for translation.
- **`frontend/`**: React application (Vite). Modern UI for document management and results.

## Deployment with Docker (Recommended)

The easiest way to run the entire stack is using Docker Compose.

1. **Configure Environment**:
   Ensure you have a `.env` file in the `backend/` directory with your API keys (e.g., `OPENROUTER_KEY`).

2. **Run with Docker Compose**:
   ```bash
   docker-compose up --build -d
   ```

3. **Access the App**:
   - Frontend: `http://localhost`
   - Backend API: `http://localhost:8000`

## Local Development (Manual)

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
2. Login with any credentials (Mock Login).
3. Drag & drop an image or PDF.
4. Click **Translate Document**.
5. View the original extracted text and the translated Nepali text.
6. Download the result as a PDF.
