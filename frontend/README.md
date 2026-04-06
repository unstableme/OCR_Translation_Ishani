# NepText Frontend: React + Vite + Custom Camera API

This is the UI for the **NepText** OCR & Translation suite. It's built for speed, responsiveness, and a premium "Smart Scanning" experience.

---

## 🌟 Key Features

- **Document Ingestion**: Drag-and-drop or browsing for images and multi-page PDFs.
- **Smart Camera Support**:
  - **Native Mobile Integration**: Uses `input capture="environment"` for superior OS-level scanning.
  - **Custom Desktop Modal**: Manually implemented `MediaDevices API` (webcam) with an interactive viewfinder for laptop users.
- **AI UI Transitions**: Features "Typewriter" effect results and dynamic loading animations.
- **Language Picker**: Contextual Himalayas language support (Tamang, Newari/Nepal Bhasa, Nepali).
- **PDF Export**: PDF generation using `jsPDF`.

---

## 🏗️ Technical Details

- **Framework**: `React (Vite)`
- **Styling**: `Vanilla CSS` (Glassmorphism, custom micro-animations).
- **Icons**: `Lucide-React`.
- **API Communication**: `Axios` with dynamic hostname detection for mobile environment testing.

---

## 🚀 Setup & Local Access

1. **Installation**:
   ```bash
   npm install
   ```

2. **Start Dev Server**:
   ```bash
   npm run dev
   ```

3. **External/Mobile Access**:
   To test on your mobile device (on the same Wi-Fi), expose the host:
   ```bash
   npm run dev -- --host
   ```
   *Then access via your laptop's Local IP (`http://192.168.1.XX:5173`).*

---

## 📂 Structure

- `src/components/Dashboard.jsx`: The core engine of the UI.
- `src/components/Dashboard.css`: Premium styling logic.
- `src/fonts/`: Devanagari font support for PDF generation.

---

## 📜 Project Overview
This frontend communicates with the **FastAPI** backend to perform heavy OCR and AI translation tasks.

---
*NepText Frontend Utility - 2026*
