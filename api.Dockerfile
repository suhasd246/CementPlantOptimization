# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies globally (not in venv)
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy application code
COPY . .

# Change ownership to avoid permission issues
RUN chmod -R 755 /app

# Expose port 8000
EXPOSE 8000

# Command to run FastAPI via Gunicorn + UvicornWorker
CMD exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} src.api.main:app
