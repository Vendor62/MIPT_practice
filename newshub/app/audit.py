import json
import logging
import os
from datetime import datetime, timezone

AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "/app/logs/audit.log")

_logger = logging.getLogger("newshub.audit")
_logger.setLevel(logging.INFO)
_logger.propagate = False

if not _logger.handlers:
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    fh = logging.FileHandler(AUDIT_LOG_PATH)
    fh.setLevel(logging.INFO)
    _logger.addHandler(fh)

def audit(event: str, **fields):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    _logger.info(json.dumps(rec, ensure_ascii=False))
