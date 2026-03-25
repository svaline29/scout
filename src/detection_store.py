"""Append-only in-memory detection log with JSON persistence."""

from __future__ import annotations

import json
import os


class DetectionStore:
    def __init__(self) -> None:
        self.detections: list[dict] = []

    def add(self, detections: list[dict]) -> None:
        self.detections.extend(detections)

    def save(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.detections, f, indent=2)

    def load(self, path: str) -> list[dict]:
        if not os.path.isfile(path):
            print(f"Warning: detection file not found: {path}")
            self.detections = []
            return []
        with open(path, encoding="utf-8") as f:
            self.detections = json.load(f)
        return self.detections
