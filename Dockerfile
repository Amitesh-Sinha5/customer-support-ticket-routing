FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ticket_router ticket_router
COPY data data
COPY scripts scripts

# Train at build time so the container starts instantly and never writes models at runtime.
RUN python scripts/train.py

EXPOSE 8000

# One process only: tickets and agent workloads are held in memory (see README limitations).
# Hosts such as Render, Railway and Cloud Run inject PORT; 8000 is the local default.
CMD ["sh", "-c", "exec uvicorn ticket_router.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
