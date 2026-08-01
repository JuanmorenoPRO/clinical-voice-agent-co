FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema mínimas (PyMuPDF trae wheels; psycopg[binary] también).
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY apps/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Código del backend + prompts + datos de ejemplo (para seed).
COPY apps/backend/app ./app
COPY prompts ./prompts
COPY data ./data
COPY seed.py ./seed.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
