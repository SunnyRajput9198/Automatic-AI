FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories for Week 1 + Week 2
RUN mkdir -p /app/sandbox && chmod 777 /app/sandbox
RUN mkdir -p /app/workspace/shared && chmod 777 /app/workspace
RUN mkdir -p /app/traces && chmod 777 /app/traces

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]