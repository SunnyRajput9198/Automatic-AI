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
RUN mkdir -p /app/sandbox && chmod 755 /app/sandbox
RUN mkdir -p /app/workspace/shared && chmod 755 /app/workspace
RUN mkdir -p /app/traces && chmod 755 /app/traces
RUN mkdir -p /app/costs && chmod 755 /app/costs
# 🔥 VERY IMPORTANT
VOLUME ["/app/workspace"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]