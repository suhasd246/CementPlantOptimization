# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard code
COPY dashboard ./dashboard
COPY run_dashboard.py .

# Expose port (for documentation only, Cloud Run uses $PORT)
EXPOSE 8080

# Run the Dash app with Gunicorn, binding to Cloud Run's $PORT
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "dashboard.main:server"]
