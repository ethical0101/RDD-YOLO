"""FastAPI application entry point.

Run (from the project root):
    .venv\\Scripts\\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import rdd_yolo  # noqa: E402

from .api.routes import router  # noqa: E402
from .core.config import get_settings  # noqa: E402
from .db.database import init_engine  # noqa: E402
from .services.model_service import model_service  # noqa: E402

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rdd_yolo.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_engine()
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    if not settings.lazy_model:
        model_service.load()
    yield


app = FastAPI(title=settings.app_name, version=rdd_yolo.__version__, lifespan=lifespan,
              description="Road damage detection (RDD2022 classes D00/D10/D20/D40), geolocation and mapping.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

settings.outputs_dir.mkdir(parents=True, exist_ok=True)
settings.experiments_dir.mkdir(parents=True, exist_ok=True)
app.mount("/files/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")
app.mount("/files/experiments", StaticFiles(directory=settings.experiments_dir), name="experiments")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}: {exc}"})


# Serve the built dashboard (frontend/dist) when present, so one process can host everything.
_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(("api/", "files/")) or full_path in ("api", "files"):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        f = (_dist / full_path).resolve()
        if full_path and f.is_file() and f.is_relative_to(_dist.resolve()):
            return FileResponse(f)
        return FileResponse(_dist / "index.html")
