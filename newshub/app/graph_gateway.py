from __future__ import annotations

import base64
import os
from typing import Any

import aiohttp
import structlog

try:
    from neo4j import AsyncGraphDatabase
except ImportError:  # pragma: no cover - optional runtime dependency in core image
    AsyncGraphDatabase = None


log = structlog.get_logger()


def _default_http_uri_from_bolt(uri: str) -> str:
    raw = str(uri or "").strip()
    if raw.startswith("bolt://"):
        return "http://" + raw[len("bolt://") :]
    if raw.startswith("neo4j://"):
        return "http://" + raw[len("neo4j://") :]
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return "http://neo4j:7474"


class GraphGateway:
    def __init__(self) -> None:
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.neo4j_http_uri = os.getenv(
            "NEO4J_HTTP_URI",
            _default_http_uri_from_bolt(self.neo4j_uri).replace(":7687", ":7474"),
        )
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "newshub_dev_neo4j")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = None
        if AsyncGraphDatabase is not None:
            self._driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_pool_size=10,
            )

    async def run_read_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        if self._driver is not None:
            return await self._run_via_driver(query, params)
        return await self._run_via_http(query, params)

    async def _run_via_driver(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        async with self._driver.session(database=self.neo4j_database) as session:
            result = await session.run(query, **params)
            rows = await result.data()
            return [dict(row) for row in rows]

    async def _run_via_http(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = f"{self.neo4j_http_uri.rstrip('/')}/db/{self.neo4j_database}/tx/commit"
        token = base64.b64encode(f"{self.neo4j_user}:{self.neo4j_password}".encode("utf-8")).decode("ascii")
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "statements": [
                {
                    "statement": query,
                    "parameters": params,
                }
            ]
        }
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                data = await response.json(content_type=None)
        errors = list(data.get("errors") or [])
        if errors:
            log.warning("graph_gateway_http_error", endpoint=endpoint, errors=errors)
            raise RuntimeError(str(errors[0]))
        results = list(data.get("results") or [])
        if not results:
            return []
        columns = list(results[0].get("columns") or [])
        rows: list[dict[str, Any]] = []
        for item in results[0].get("data") or []:
            row_values = list(item.get("row") or [])
            rows.append(dict(zip(columns, row_values)))
        return rows


graph_gateway = GraphGateway()
