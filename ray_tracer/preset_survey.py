"""Survey and collection helpers for preset YAML assets.

This module inventories YAML presets and config files so the unified scene I/O
layer can align with existing repository layout and schema conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PresetFileRecord:
    path: str
    category: str
    top_level_keys: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PresetSurvey:
    root: str
    categories: dict[str, list[PresetFileRecord]] = field(default_factory=dict)


class PresetSurveyCoordinator:
    """Collect and export YAML preset layout and content snapshots."""

    def __init__(self, workspace_root: str | Path = ".") -> None:
        self.workspace_root = Path(workspace_root)

    def survey(self) -> PresetSurvey:
        files = self._collect_yaml_files()
        categories: dict[str, list[PresetFileRecord]] = {}
        for path in files:
            rel = path.relative_to(self.workspace_root).as_posix()
            category = self._category_for(rel)
            payload = self._load_yaml(path)
            top_keys = list(payload.keys()) if isinstance(payload, dict) else []
            rec = PresetFileRecord(
                path=rel,
                category=category,
                top_level_keys=top_keys,
                payload=payload if isinstance(payload, dict) else {},
            )
            categories.setdefault(category, []).append(rec)

        for bucket in categories.values():
            bucket.sort(key=lambda item: item.path)

        return PresetSurvey(root=self.workspace_root.as_posix(), categories=categories)

    def export_json(self, path: str | Path) -> PresetSurvey:
        survey = self.survey()
        out = {
            "root": survey.root,
            "categories": {
                category: [
                    {
                        "path": rec.path,
                        "category": rec.category,
                        "top_level_keys": rec.top_level_keys,
                        "payload": rec.payload,
                    }
                    for rec in records
                ]
                for category, records in survey.categories.items()
            },
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return survey

    def _collect_yaml_files(self) -> list[Path]:
        roots = [self.workspace_root / "configs", self.workspace_root / "presets"]
        files: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            files.extend(root.rglob("*.yaml"))
            files.extend(root.rglob("*.yml"))
        return sorted(set(files))

    @staticmethod
    def _category_for(relative_path: str) -> str:
        parts = relative_path.split("/")
        if len(parts) < 2:
            return "uncategorized"
        return "/".join(parts[:2])

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        try:
            import yaml
        except Exception:
            return {}

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:
            return {}

        return data if isinstance(data, dict) else {}
