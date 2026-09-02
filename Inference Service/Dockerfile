# Use Python 3.9 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for numpy/scipy)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .
COPY models.py .

# Create models directory
RUN mkdir -p models

# Copy model files
COPY models/svd_model_final.pkl models/
# Optional: Copy ALS fallback model if available (comment out if not using)
COPY models/als_bundle.pkl models/

# Expose port 8082
EXPOSE 8082

# Set environment variables
ENV FLASK_APP=server.py
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8082/')" || exit 1

# Run the application
CMD ["python", "server.py"]