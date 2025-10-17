# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard code and the entrypoint
COPY dashboard ./dashboard
COPY run_dashboard.py .

# Expose port 8050 (for documentation/clarity)
EXPOSE 8050

# Run the Dash app via Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT dashboard.main:app