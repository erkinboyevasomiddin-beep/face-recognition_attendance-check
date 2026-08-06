FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system attendance && adduser --system --ingroup attendance attendance

COPY pyproject.toml constraints-python311.txt README.md LICENSE .env.example ./
COPY backend ./backend
COPY recognition/__init__.py ./recognition/__init__.py
COPY sample_data ./sample_data
RUN python -m pip install "pip>=26.1.2" && \
    python -m pip install -c constraints-python311.txt .

RUN mkdir -p /app/backend/data && chown -R attendance:attendance /app
USER attendance

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
