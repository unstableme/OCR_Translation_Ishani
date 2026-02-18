# NepText

A professional AI-powered tool for OCR extraction and translation. **NepText** seamlessly extracts text from images and PDF documents and translates them into Nepali. Built with **FastAPI** (Backend) and **React** (Frontend).

## Project Structure

- **`backend/`**: Python FastAPI server. Handles file uploads, OCR processing, and translation.
- **`frontend/`**: React application (Vite). Modern UI for document management and results.
- **`.github/workflows/`**: CI/CD pipelines to build and push Docker images to Docker Hub.

## Environment Setup

The system requires environment variables to function (AI keys, database URLs, etc.).

1. **Locate the Template**: See `.env.example` in the root directory.
2. **Create your `.env`**: Copy `.env.example` to a new file named `.env` in the project root.
   ```bash
   cp .env.example .env
   ```
3. **Fill in the values**: Open `.env` and add your `OPENROUTER_KEY`. 


## Deployment with Docker (Recommended)

The easiest way to run the entire stack is using Docker Compose.

1. **Configure Environment**: Ensure your root `.env` is ready as described above.
2. **Run with Docker Compose**:
   ```bash
   # Build locally and start
   docker-compose up --build -d
   ```
   *Note: On a production server, you only need `docker-compose.yml` and `.env`. Running `docker-compose pull` will download the pre-built images from Docker Hub.*

3. **Access the App**:
   - Frontend: `http://localhost:5173`
   - Backend API: `http://localhost:8000`

## Local Development (Manual)

### 1. Backend Setup
Navigate to the `backend` directory:
```bash
cd backend
python -m venv venv
# Activate venv (Windows: .\venv\Scripts\activate | Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload
```
*The backend uses `find_dotenv()` to automatically find the `.env` file in the root directory.*

### 2. Frontend Setup
Navigate to the `frontend` directory:
```bash
cd frontend
npm install
npm run dev
```
The frontend will run at `http://localhost:5173`.

## CI/CD
This project uses GitHub Actions to automatically build and push Docker images to Docker Hub on every push to the `main` branch.
- Backend: `unstableme02/ishani_dms-backend`
- Frontend: `unstableme02/ishani_dms-frontend`
