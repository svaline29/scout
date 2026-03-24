from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cuvslam

_IDENTITY_GUESS_POSE: cuvslam.Pose = cuvslam.Pose(
    rotation=[0, 0, 0, 1],
    translation=[0, 0, 0],
)


def _validate_map_name(map_name: str) -> None:
    if not map_name or map_name != Path(map_name).name:
        raise ValueError(f"invalid map_name: {map_name!r}")
    if ".." in map_name or "/" in map_name or "\\" in map_name:
        raise ValueError(f"invalid map_name: {map_name!r}")


class MapManager:
    """Persists and restores cuVSLAM maps under a fixed root directory."""

    maps_dir: Path
    last_loaded_metadata: dict[str, Any] | None
    last_localization_pose: Any | None

    def __init__(self, maps_dir: str) -> None:
        self.maps_dir = Path(maps_dir)
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.last_loaded_metadata = None
        self.last_localization_pose = None

    def _map_path(self, map_name: str) -> Path:
        _validate_map_name(map_name)
        return self.maps_dir / map_name

    def save(self, tracker: Any, map_name: str, metadata: dict[str, Any]) -> bool:
        """Save a cuVSLAM map under ``maps_dir / map_name`` and write ``metadata.json`` on success."""
        map_path = self._map_path(map_name)
        map_path.mkdir(parents=True, exist_ok=True)

        done: threading.Event = threading.Event()
        success_flag: list[bool] = [False]

        def _on_save(success: bool) -> None:
            success_flag[0] = success
            done.set()

        tracker.save_map(os.fspath(map_path), _on_save)
        if not done.wait(timeout=30):
            print("Warning: operation timed out")
            return False

        if not success_flag[0]:
            return False

        created_at: str = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "map_name": map_name,
            "created_at": created_at,
            **metadata,
        }
        meta_file: Path = map_path / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True

    def load(
        self,
        tracker: Any,
        map_name: str,
        first_frame: Any,
        settings: Any,
    ) -> bool:
        """
        Load a map and run ``localize_in_map`` with an identity pose guess.

        On success, sets ``last_loaded_metadata`` and ``last_localization_pose``.
        """
        map_path: Path = self._map_path(map_name)
        meta_file: Path = map_path / "metadata.json"

        if not map_path.is_dir() or not meta_file.is_file():
            self.last_loaded_metadata = None
            self.last_localization_pose = None
            return False

        with open(meta_file, encoding="utf-8") as f:
            loaded: dict[str, Any] = json.load(f)

        self.last_loaded_metadata = loaded

        done: threading.Event = threading.Event()
        ok: list[bool] = [False]

        def _on_localize(pose: Any, error_message: str) -> None:
            self.last_localization_pose = pose
            if error_message:
                ok[0] = False
            else:
                ok[0] = pose is not None
            done.set()

        tracker.localize_in_map(
            os.fspath(map_path),
            _IDENTITY_GUESS_POSE,
            first_frame,
            settings,
            _on_localize,
        )
        if not done.wait(timeout=30):
            print("Warning: operation timed out")
            self.last_localization_pose = None
            return False
        return ok[0]

    def list_maps(self) -> list[dict[str, Any]]:
        """Return one entry per saved map: ``map_name`` plus fields from ``metadata.json``."""
        out: list[dict[str, Any]] = []
        if not self.maps_dir.is_dir():
            return out

        for entry in sorted(self.maps_dir.iterdir()):
            if not entry.is_dir():
                continue
            name: str = entry.name
            meta_path: Path = entry / "metadata.json"
            if not meta_path.is_file():
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            row: dict[str, Any] = {**data, "map_name": name}
            out.append(row)
        return out
