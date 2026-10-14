# Production Dockerfile for GCN Serving API
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    fastapi \
    uvicorn \
    pydantic \
    requests
    


COPY task1 /app/task1
COPY task4 /app/task4


ENV PYTHONPATH=/app

# Expose FastAPI serving port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn production server
CMD ["uvicorn", "task4.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
