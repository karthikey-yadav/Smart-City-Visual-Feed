FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py feature_extractor.py ./
COPY quality_model_best.pt ./

# Persisted SQLite DB lives in a mounted volume (see docker-compose.yml)
ENV MODEL_PATH=/app/quality_model_best.pt
ENV DB_PATH=/app/data/quality_results.db
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
