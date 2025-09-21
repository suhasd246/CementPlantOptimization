# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dashboard code and the APIClient helper
COPY dashboard ./dashboard
COPY run_dashboard.py .

# Expose the port the app runs on (Dash default is 8050)
EXPOSE 8050

# Command to run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "dashboard.main:server"]