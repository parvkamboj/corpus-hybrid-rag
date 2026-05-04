from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GoldenSample:
    query: str
    ground_truth: str
    doc_ids: list[str] = field(default_factory=list)


@dataclass
class GoldenDataset:
    name: str
    samples: list[GoldenSample]

    def __len__(self) -> int:
        return len(self.samples)


def load_dataset(path: Path) -> GoldenDataset:
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}")

    name: str = raw.get("name", path.stem)
    raw_samples: list[Any] = raw.get("samples", [])
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"'samples' must be a non-empty list in {path}")

    samples: list[GoldenSample] = []
    for i, s in enumerate(raw_samples):
        if not isinstance(s, dict):
            raise ValueError(f"Sample {i} is not a mapping")
        query = s.get("query", "")
        ground_truth = s.get("ground_truth", "")
        if not query or not ground_truth:
            raise ValueError(f"Sample {i} missing required 'query' or 'ground_truth'")
        samples.append(
            GoldenSample(
                query=str(query),
                ground_truth=str(ground_truth),
                doc_ids=[str(d) for d in s.get("doc_ids", [])],
            )
        )

    return GoldenDataset(name=name, samples=samples)
