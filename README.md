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
... [previous instructions] ...

