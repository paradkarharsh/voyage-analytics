# ---- Flight-price / gender / recommendation API ----
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    VOYAGE_MODEL_DIR=/app/models
WORKDIR /app

# libgomp is required by xgboost / scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/

RUN useradd -r -u 1001 appuser && chown -R appuser /app
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fs http://localhost:5000/health || exit 1

# 1 worker + threads keeps memory low (models are loaded once per worker)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "60", "api.app:get_app()"]
