#легаси ембеддинги - отключено

import os
import structlog
import aiohttp
from datetime import datetime

log = structlog.get_logger()

EMBEDDINGS_SERVICE_URL = os.getenv("EMBEDDINGS_SERVICE_URL", "").strip()
EMBEDDINGS_TIMEOUT = int(os.getenv("EMBEDDINGS_TIMEOUT", "20"))
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY", "").strip()


class EmbeddingProcessor:
    def __init__(self):
        if not EMBEDDINGS_SERVICE_URL:
            log.warning("EMBEDDINGS_SERVICE_URL not set, embeddings disabled")

    async def process_post(self, post_id: int, raw_text: str, timestamp: datetime):
        if not EMBEDDINGS_SERVICE_URL:
            return None

        payload = {
            "post_id": post_id,
            "text": raw_text,
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else None,
        }
        headers = {"Content-Type": "application/json"}
        if EMBEDDINGS_API_KEY:
            headers["Authorization"] = f"Bearer {EMBEDDINGS_API_KEY}"

        try:
            timeout = aiohttp.ClientTimeout(total=EMBEDDINGS_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    EMBEDDINGS_SERVICE_URL,
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.error("Embeddings service error", status=resp.status, body=body)
                        return None
                    data = await resp.json()
                    embedding = data.get("embedding")
                    if not isinstance(embedding, list):
                        log.error("Embeddings service returned invalid payload", data=data)
                        return None
                    return embedding
        except Exception as e:
            log.error("Embeddings request failed", error=str(e))
            return None
