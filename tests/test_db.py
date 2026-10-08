"""Database layer tests on an isolated SQLite file."""
from __future__ import annotations

from sqlalchemy import select


def test_crud_cascade_and_nearby(tmp_path):
    from backend.app.db import database
    from backend.app.db.models import Detection, Inference
    from backend.app.services.detections import count_nearby

    database.init_engine(f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    Session = database.session_factory()
    with Session() as db:
        inf = Inference(kind="image", model_version="test", conf_threshold=0.25, latitude=12.9692, longitude=79.1559,
                        location_source="manual")
        db.add(inf)
        db.flush()
        for dx in (0.0, 0.00005, 0.01):  # 0 m, ~5.5 m, ~1.1 km
            db.add(Detection(inference_id=inf.id, class_code="D40", class_name="Pothole", confidence=0.8, x1=0, y1=0,
                             x2=10, y2=10, severity="HIGH", severity_score=70, latitude=12.9692 + dx,
                             longitude=79.1559, location_source="manual", model_version="test"))
        db.commit()
        assert count_nearby(db, 12.9692, 79.1559, "D40") == 2  # within 15 m
        assert count_nearby(db, 12.9692, 79.1559, "D00") == 0
        assert count_nearby(db, None, None, "D40") == 0
        db.delete(inf)
        db.commit()
        assert db.scalars(select(Detection)).all() == []  # cascade
