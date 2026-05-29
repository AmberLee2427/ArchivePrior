from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


@dataclass(frozen=True)
class QueryProvenance:
    timestamp: str
    row_count: int
    source_url: str
    cache_path: str
    cache_hit: bool
    table: str


class ArchiveClient:
    """Fetch and cache tables from the NASA Exoplanet Archive."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        base_url: str = DEFAULT_TAP_URL,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.timeout = float(timeout)
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "archiveprior")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, table: str, query: str) -> Path:
        digest = sha256(f"{self.base_url}|{query}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{table}_{digest}.csv"

    def fetch_pscomppars(self, refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch the `pscomppars` table as a cached pandas DataFrame."""
        table = "pscomppars"
        query = f"select * from {table}"
        cache_path = self._cache_path(table, query)

        if cache_path.exists() and not refresh:
            frame = pd.read_csv(cache_path)
            provenance = QueryProvenance(
                timestamp=datetime.now(timezone.utc).isoformat(),
                row_count=int(len(frame)),
                source_url=self.base_url,
                cache_path=str(cache_path),
                cache_hit=True,
                table=table,
            )
            return frame, provenance.__dict__

        response = requests.get(
            self.base_url,
            params={"query": query, "format": "csv"},
            timeout=self.timeout,
        )
        response.raise_for_status()

        cache_path.write_text(response.text, encoding="utf-8")
        frame = pd.read_csv(cache_path)
        provenance = QueryProvenance(
            timestamp=datetime.now(timezone.utc).isoformat(),
            row_count=int(len(frame)),
            source_url=response.url,
            cache_path=str(cache_path),
            cache_hit=False,
            table=table,
        )
        return frame, provenance.__dict__