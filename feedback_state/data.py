"""JSONL dataset loading for evaluation and the checks in analysis/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlDataset:
    """Load a JSONL event stream while preserving its on-disk order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]
