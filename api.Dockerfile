# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Create a non-root user (good practice for Cloud Run)
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Create a virtual environment and ensure PATH points to it
ENV VIRTUAL_ENV=/app/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8000
EXPOSE 8000

# Use Gunicorn + Uvicorn worker to serve the FastAPI app
CMD exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} src.api.main:app
