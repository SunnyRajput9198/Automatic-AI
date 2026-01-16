FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create sandbox directory
RUN mkdir -p /app/sandbox && chmod 777 /app/sandbox

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]