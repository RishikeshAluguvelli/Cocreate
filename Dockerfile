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
# run as an unprivileged user (uid 1000, which is what Hugging Face Spaces use); everything under /srv is writable by it
RUN useradd -m -u 1000 app
WORKDIR /srv
COPY backend/pyproject.toml backend/uv.lock ./backend/
RUN cd backend && uv sync --frozen --no-dev --no-install-project
COPY backend ./backend
COPY data/customer-requests.json ./data/customer-requests.json
COPY --from=web /web/dist ./backend/app/static
RUN mkdir -p /srv/data && chown -R app:app /srv
USER app
WORKDIR /srv/backend
ENV PATH="/srv/backend/.venv/bin:$PATH" DATA_DIR=/srv/data PORT=8000 HOME=/home/app
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 120"]
