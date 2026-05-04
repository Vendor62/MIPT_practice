from __future__ import annotations

"""
Compact DeepSeek baseline (OpenAI-compatible chat completions).

This is a reference implementation used to describe the protocol.
It is intentionally decoupled from private data sources.
"""

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class DeepSeekConfig:
    api_url: str
    api_key: str
    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 400


def load_deepseek_config_from_env() -> DeepSeekConfig:
    return DeepSeekConfig(
        api_url=os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "400")),
    )


def deepseek_choose_story(
    *,
    cfg: DeepSeekConfig,
    post_text: str,
    candidate_stories: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Returns JSON:
      { decision: "attach"|"new_story", selected_story_id: str|null, confidence: number, short_reason: str }

    IMPORTANT: gold labels must never be included in the prompt.
    """
    if not cfg.api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing (env)")

    system = (
        "Ты помощник новостного сторитрекинга.\n"
        "Выбери: attach к одному candidate story_id или new_story.\n"
        "Не объединяй по общим сущностям.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "Схема: {\"decision\":\"attach\"|\"new_story\",\"selected_story_id\":string|null,\"confidence\":number,\"short_reason\":string}\n"
    )
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {"post_text": post_text[:1500], "candidate_stories": candidate_stories[:8]},
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(
        cfg.api_url,
        headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)

