# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only API-related code
COPY src ./src
COPY artifacts ./artifacts

# Expose port (for documentation only, Cloud Run uses $PORT)
EXPOSE 8080

# Run the app with Gunicorn, binding to Cloud Run's $PORT
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:${PORT}", "src.api.main:app"]
