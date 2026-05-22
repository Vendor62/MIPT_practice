from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

STORYLINE_PAIR_SCORER_ENABLED = os.getenv("STORYLINE_PAIR_SCORER_ENABLED", "1") != "0"
STORYLINE_PAIR_SCORER_MODEL_DIR = os.getenv(
    "STORYLINE_PAIR_SCORER_MODEL_DIR",
    "/app/artifacts/model_runs/storyline_pair_scorer_catboost_20260322T181414Z",
)

_model = None
_feature_cols: list[str] | None = None
_cat_features: set[str] = set()
_load_error = ""


def _to_num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _load():
    global _model, _feature_cols, _cat_features, _load_error
    if _model is not None or _load_error:
        return
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        _load_error = f"catboost_import_error:{exc}"
        log.warning("storyline_pair_scorer_disabled", reason=_load_error)
        return

    model_dir = Path(STORYLINE_PAIR_SCORER_MODEL_DIR)
    model_path = model_dir / "storyline_pair_scorer_catboost.cbm"
    meta_path = model_dir / "storyline_pair_scorer_feature_cols.json"
    if not model_path.exists() or not meta_path.exists():
        fallback_dir = Path("/app/artifacts/model_runs")
        fallback_model = fallback_dir / "storyline_pair_scorer_catboost.cbm"
        fallback_meta = fallback_dir / "storyline_pair_scorer_feature_cols.json"
        if fallback_model.exists() and fallback_meta.exists():
            model_dir = fallback_dir
            model_path = fallback_model
            meta_path = fallback_meta
        else:
            _load_error = "model_or_meta_not_found"
            log.warning(
                "storyline_pair_scorer_disabled",
                reason=_load_error,
                model_dir=str(model_dir),
            )
            return

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_cols = list(meta.get("feature_cols") or [])
        cat_features = set(meta.get("cat_features") or [])
        m = CatBoostClassifier()
        m.load_model(str(model_path))
    except Exception as exc:
        _load_error = f"model_load_error:{exc}"
        log.warning("storyline_pair_scorer_disabled", reason=_load_error)
        return

    if not feature_cols:
        _load_error = "empty_feature_cols"
        log.warning("storyline_pair_scorer_disabled", reason=_load_error)
        return

    _model = m
    _feature_cols = feature_cols
    _cat_features = cat_features
    log.info("storyline_pair_scorer_enabled", model_dir=str(model_dir))


def scorer_status() -> dict[str, Any]:
    if not STORYLINE_PAIR_SCORER_ENABLED:
        return {"enabled": False, "ready": False, "reason": "feature_flag_off"}
    _load()
    return {"enabled": True, "ready": _model is not None, "reason": _load_error}


def score_candidates(feature_rows: list[dict[str, Any]]) -> list[float]:
    if not STORYLINE_PAIR_SCORER_ENABLED:
        return []
    _load()
    if _model is None or _feature_cols is None:
        return []
    if not feature_rows:
        return []

    X = []
    for row in feature_rows:
        out = []
        for col in _feature_cols:
            if col in _cat_features:
                out.append(str(row.get(col, "") or ""))
            else:
                out.append(_to_num(row.get(col)))
        X.append(out)

    try:
        proba = _model.predict_proba(X)
        return [float(p[1]) for p in proba]
    except Exception as exc:
        log.warning("storyline_pair_scorer_predict_failed", error=str(exc))
        return []
