# Use an official Python runtime as a parent image (slim for smaller size)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# gcc and libpq-dev are needed for psycopg2
# libgl1 and libglib2.0-0 for OpenCV
# tesseract-ocr(-nep) for OCR engine
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-nep \
    libtesseract-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create uploads and debug directories
RUN mkdir -p uploads

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
