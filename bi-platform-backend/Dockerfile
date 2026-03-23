FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (removed postgresql-client)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Create directories for file-based data storage
RUN mkdir -p /app/data/datasets /app/data/models /app/uploads
# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
