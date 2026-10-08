"""Shared pytest fixtures.

Tests run against an isolated temporary database/outputs directory and use
CPU inference so they never interfere with a training run on the GPU.

Model-dependent tests need *a* real checkpoint. They use, in order:
  1. $RDD_TEST_WEIGHTS
  2. models/weights/rdd_yolo26n_best.pt / baseline_yolo26n_best.pt (trained models)
  3. experiments/smoke_rdd/weights/best.pt (tiny smoke-test run, see docs/training.md)
and are skipped if none exists. The smoke checkpoint is only used to exercise
the code paths; its predictions are not meaningful.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_DIRS = [ROOT / "dataset" / "processed" / "rdd2022_yolo" / "images" / "test",
               ROOT / "dataset" / "processed" / "smoke" / "images" / "test"]


def find_weights() -> Path | None:
    cands = [os.environ.get("RDD_TEST_WEIGHTS")] if os.environ.get("RDD_TEST_WEIGHTS") else []
    cands += [ROOT / "models/weights/rdd_yolo26n_ext_best.pt", ROOT / "models/weights/rdd_yolo26n_best.pt", ROOT / "models/weights/baseline_yolo26n_best.pt",
              ROOT / "experiments/smoke_rdd/weights/best.pt"]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def sample_images(n: int = 3) -> list[Path]:
    for d in SAMPLE_DIRS:
        if d.exists():
            imgs = sorted(d.glob("*.jpg"))[:n]
            if imgs:
                return imgs
    return []


@pytest.fixture(scope="session")
def weights() -> Path:
    w = find_weights()
    if w is None:
        pytest.skip("No checkpoint available (train a model or run the smoke training first)")
    return w


@pytest.fixture(scope="session")
def sample_image() -> Path:
    imgs = sample_images(1)
    if not imgs:
        pytest.skip("No prepared dataset images available")
    return imgs[0]


@pytest.fixture(scope="session")
def app_env(tmp_path_factory, weights):
    tmp = tmp_path_factory.mktemp("rdd")
    os.environ["RDD_DATABASE_URL"] = f"sqlite:///{(tmp / 'test.db').as_posix()}"
    os.environ["RDD_OUTPUTS_DIR"] = str(tmp / "outputs")
    os.environ["RDD_MODEL_WEIGHTS"] = str(weights)
    os.environ["RDD_MODEL_DEVICE"] = "cpu"
    os.environ["RDD_DEFAULT_CONFIDENCE"] = "0.25"
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    return tmp


@pytest.fixture(scope="session")
def client(app_env):
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as c:
        yield c
