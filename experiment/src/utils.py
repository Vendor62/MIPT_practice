from __future__ import annotations

"""
Вспомогательные функции для чтения CSV/JSON и стабильной сортировки постов по времени.
"""

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_ts(s: str) -> datetime | None:
    t = (s or "").strip()
    if not t:
        return None
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00").replace(" ", "T"))
    except Exception:
        return None


def stable_sorted_posts(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    items = []
    for r in rows:
        pid = int(r["post_id"])
        ts = parse_ts(r.get("published_at", "") or "")
        items.append((ts or datetime.min, pid, r))
    items.sort(key=lambda t: (t[0], t[1]))
    return [r for _ts, _pid, r in items]


@dataclass(frozen=True)
class Split:
    train: set[int]
    dev: set[int]
    test: set[int]
