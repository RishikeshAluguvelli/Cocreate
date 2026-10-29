# ---- frontend ----
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend ./
RUN npm run build

# ---- backend ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /srv
COPY backend/pyproject.toml backend/uv.lock ./backend/
RUN cd backend && uv sync --frozen --no-dev --no-install-project
COPY backend ./backend
COPY data/customer-requests.json ./data/customer-requests.json
COPY --from=web /web/dist ./backend/app/static
WORKDIR /srv/backend
ENV PATH="/srv/backend/.venv/bin:$PATH" DATA_DIR=/srv/data PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 120"]
