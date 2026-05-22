from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

RELATION_SCORER_ENABLED = os.getenv("RELATION_SCORER_ENABLED", "1") != "0"
RELATION_SCORER_THRESHOLD = float(os.getenv("RELATION_SCORER_THRESHOLD", "0.52"))

_DEFAULT_MODEL_DIR_CANDIDATES = [
    os.getenv("RELATION_SCORER_MODEL_DIR", "").strip(),
    "./artifacts/model_runs/relation_scorer_catboost_20260321T140313Z",
    "/app/embeddings_service/tmp_relation_training/model_runs/relation_scorer_catboost_20260321T140313Z",
    "/app/artifacts/model_runs/relation_scorer_catboost_20260321T140313Z",
]

ORG_KEYWORDS = {
    "банк",
    "компания",
    "корпорац",
    "холдинг",
    "госкорпорац",
    "министер",
    "ведомств",
    "правительств",
    "администрац",
    "парламент",
    "дума",
    "совет",
    "университет",
    "институт",
    "центр",
    "служба",
    "комитет",
    "агентств",
    "фонд",
    "армия",
    "командование",
    "партия",
    "суд",
    "прокуратур",
}

LOCATION_KEYWORDS = {
    "город",
    "область",
    "край",
    "район",
    "республик",
    "страна",
    "пролив",
    "регион",
    "округ",
    "остров",
    "посел",
    "провинц",
    "столица",
    "улица",
    "море",
    "океан",
    "континент",
}

RELATION_DIRECTION_PRIORS = {
    "employer": ("person", "org"),
    "position_held": ("person", "org"),
    "member_of": ("person", "org"),
    "member_of_political_party": ("person", "org"),
    "founded_by": ("org", "person"),
    "chairperson": ("person", "org"),
    "director": ("person", "org"),
    "born_in": ("person", "location"),
    "place_of_death": ("person", "location"),
    "citizen_of": ("person", "location"),
    "located_in": ("org", "location"),
    "country_of_origin": ("org", "location"),
}

_RUNTIME_SCORER = None
_RUNTIME_SCORER_INIT_DONE = False


def _to_num(value: Any) -> float:
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    if s.lower() == "true":
        return 1.0
    if s.lower() == "false":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text or "")


def _has_digit(text: str) -> int:
    return 1 if re.search(r"\d", text or "") else 0


def _has_latin(text: str) -> int:
    return 1 if re.search(r"[A-Za-z]", text or "") else 0


def _has_cyrillic(text: str) -> int:
    return 1 if re.search(r"[А-Яа-яЁё]", text or "") else 0


def _guess_entity_type(surface: str) -> str:
    s = (surface or "").strip()
    if not s:
        return "other"
    s_low = s.lower()

    person_like_tokens = re.findall(r"\b[А-ЯЁA-Z][а-яёa-z-]{2,}\b", s)
    if len(person_like_tokens) >= 2 and not any(k in s_low for k in ORG_KEYWORDS):
        return "person"

    if any(k in s_low for k in ORG_KEYWORDS):
        return "org"
    if any(k in s_low for k in LOCATION_KEYWORDS):
        return "location"
    return "other"


def _build_derived_features(row: dict[str, Any]) -> dict[str, Any]:
    head = str(row.get("head", "") or "")
    tail = str(row.get("tail", "") or "")
    relation = str(row.get("relation", "") or "").strip().lower()

    head_tokens = _tokenize(head)
    tail_tokens = _tokenize(tail)
    head_set = {t.lower() for t in head_tokens}
    tail_set = {t.lower() for t in tail_tokens}
    union = head_set | tail_set
    inter = head_set & tail_set
    jaccard = (len(inter) / len(union)) if union else 0.0

    head_type = _guess_entity_type(head)
    tail_type = _guess_entity_type(tail)
    expected = RELATION_DIRECTION_PRIORS.get(relation)

    return {
        "head_token_count": len(head_tokens),
        "tail_token_count": len(tail_tokens),
        "head_char_len": len(head),
        "tail_char_len": len(tail),
        "token_count_diff_abs": abs(len(head_tokens) - len(tail_tokens)),
        "char_len_diff_abs": abs(len(head) - len(tail)),
        "head_tail_token_jaccard": round(jaccard, 6),
        "head_has_digit": _has_digit(head),
        "tail_has_digit": _has_digit(tail),
        "head_has_latin": _has_latin(head),
        "tail_has_latin": _has_latin(tail),
        "head_has_cyrillic": _has_cyrillic(head),
        "tail_has_cyrillic": _has_cyrillic(tail),
        "head_type_guess": head_type,
        "tail_type_guess": tail_type,
        "type_pair_guess": f"{head_type}->{tail_type}",
        "same_type_guess": 1 if head_type == tail_type else 0,
        "relation_has_direction_prior": 1 if expected else 0,
        "direction_match_expected": 1 if expected and (head_type, tail_type) == expected else 0,
        "direction_reverse_expected": 1 if expected and (tail_type, head_type) == expected else 0,
        "is_person_head_guess": 1 if head_type == "person" else 0,
        "is_person_tail_guess": 1 if tail_type == "person" else 0,
        "is_org_head_guess": 1 if head_type == "org" else 0,
        "is_org_tail_guess": 1 if tail_type == "org" else 0,
        "is_location_head_guess": 1 if head_type == "location" else 0,
        "is_location_tail_guess": 1 if tail_type == "location" else 0,
    }


def _feature_row(row: dict[str, Any], feature_cols: list[str], cat_feature_names: set[str]) -> list[Any]:
    full = dict(row)
    full.update(_build_derived_features(row))
    out: list[Any] = []
    for col in feature_cols:
        if col in cat_feature_names:
            out.append(str(full.get(col, "") or ""))
        else:
            out.append(_to_num(full.get(col)))
    return out


def _resolve_model_dir() -> Path | None:
    for candidate in _DEFAULT_MODEL_DIR_CANDIDATES:
        if not candidate:
            continue
        model_dir = Path(candidate)
        if (
            model_dir / "relation_scorer_catboost.cbm"
        ).exists() and (model_dir / "relation_scorer_feature_cols.json").exists():
            return model_dir
    return None


class RuntimeRelationScorer:
    def __init__(self, model_dir: Path):
        from catboost import CatBoostClassifier

        meta = json.loads((model_dir / "relation_scorer_feature_cols.json").read_text(encoding="utf-8"))
        self.feature_cols = list(meta["feature_cols"])
        self.cat_feature_names = set(meta.get("cat_features", []))
        self.cat_idx = [i for i, c in enumerate(self.feature_cols) if c in self.cat_feature_names]
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_dir / "relation_scorer_catboost.cbm"))
        self.model_dir = str(model_dir)

    def score(self, relation_row: dict[str, Any]) -> float:
        from catboost import Pool

        row = _feature_row(relation_row, self.feature_cols, self.cat_feature_names)
        pool = Pool([row], cat_features=self.cat_idx)
        return float(self.model.predict_proba(pool)[0][1])


def _get_runtime_scorer() -> RuntimeRelationScorer | None:
    global _RUNTIME_SCORER, _RUNTIME_SCORER_INIT_DONE
    if _RUNTIME_SCORER_INIT_DONE:
        return _RUNTIME_SCORER
    _RUNTIME_SCORER_INIT_DONE = True

    if not RELATION_SCORER_ENABLED:
        log.info("relation_model_gate_disabled", reason="env_disabled")
        return None

    model_dir = _resolve_model_dir()
    if model_dir is None:
        log.warning("relation_model_gate_disabled", reason="model_dir_not_found")
        return None

    try:
        _RUNTIME_SCORER = RuntimeRelationScorer(model_dir)
        log.info(
            "relation_model_gate_enabled",
            model_dir=str(model_dir),
            threshold=RELATION_SCORER_THRESHOLD,
        )
    except Exception as exc:
        _RUNTIME_SCORER = None
        log.warning(
            "relation_model_gate_disabled",
            reason="scorer_init_failed",
            error=str(exc),
        )
    return _RUNTIME_SCORER


def apply_relation_model_gate(
    relations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scorer = _get_runtime_scorer()
    meta = {
        "enabled": bool(scorer),
        "threshold": RELATION_SCORER_THRESHOLD,
        "model_dir": getattr(scorer, "model_dir", ""),
        "input_relations": len(relations),
        "accepted_relations": 0,
        "rejected_relations": 0,
    }
    if not relations or scorer is None:
        meta["accepted_relations"] = len(relations)
        return relations, [], meta

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rel in relations:
        proba = scorer.score(rel)
        rel_item = dict(rel)
        rel_item["model_proba_accept"] = round(proba, 6)
        rel_item["model_decision"] = "accept" if proba >= RELATION_SCORER_THRESHOLD else "reject"
        rel_item["model_threshold"] = RELATION_SCORER_THRESHOLD
        if rel_item["model_decision"] == "accept":
            accepted.append(rel_item)
            continue
        rej = dict(rel_item)
        rej["reason"] = "model_reject_threshold"
        rej["reason_group"] = "model_gate"
        rej["write_to_graph"] = False
        rejected.append(rej)

    meta["accepted_relations"] = len(accepted)
    meta["rejected_relations"] = len(rejected)
    return accepted, rejected, meta

