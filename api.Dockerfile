# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (models, source, etc.)
COPY . .

# Expose port 8000 (for documentation/clarity)
EXPOSE 8000

# Run the FastAPI app via Gunicorn + UvicornWorker
# Using shell-form so $PORT will expand properly
CMD exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} src.api.main:app