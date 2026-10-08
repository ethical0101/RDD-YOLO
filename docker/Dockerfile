# Multi-stage image: builds the dashboard, then serves API + dashboard from one
# FastAPI process on port 8000. CPU PyTorch by default (portable); for NVIDIA GPUs
# build with  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130
# and run with the NVIDIA Container Toolkit (see docker-compose.yml).
#
# NOTE: written for convenience; not exercised in the development environment
# (Docker daemon unavailable there). The native Windows scripts are the tested path.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url ${TORCH_INDEX}
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY rdd_yolo/ rdd_yolo/
COPY backend/ backend/
COPY models/architectures/ models/architectures/
COPY --from=web /web/dist frontend/dist
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
