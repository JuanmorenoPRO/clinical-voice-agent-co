FROM python:3.12-slim

WORKDIR /app

# Deps de voz (Pipecat/aiortc) solo si INSTALL_VOICE=true. Por defecto OFF para que
# el modo texto arranque rápido y fiable (gate G2).
ARG INSTALL_VOICE=false

# Dependencias del sistema mínimas (PyMuPDF trae wheels; psycopg[binary] también).
# libopus0 lo necesita aiortc/av para el audio del modo de voz.
RUN apt-get update && apt-get install -y --no-install-recommends \
    $( [ "$INSTALL_VOICE" = "true" ] && echo "libopus0" ) \
    && rm -rf /var/lib/apt/lists/*

COPY apps/backend/requirements.txt ./requirements.txt
COPY apps/backend/requirements-voice.txt ./requirements-voice.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_VOICE" = "true" ]; then \
         pip install --no-cache-dir -r requirements-voice.txt ; \
       fi

# Código del backend + prompts + datos de ejemplo (para seed).
COPY apps/backend/app ./app
COPY prompts ./prompts
COPY data ./data
COPY seed.py ./seed.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
