"""Model export utilities for factory sign detection."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd


EXPORT_EXTENSIONS = {
    "onnx": ".onnx",
    "engine": ".engine",
}


def export_final_model(
    weights_path: Path,
    output_dir: Path,
    formats: list[str],
    imgsz: int,
    device: str | int,
) -> pd.DataFrame:
    """Export final YOLO weights to deployment formats.

    Export support varies by machine and Ultralytics install. Each requested
    format is attempted independently so a TensorRT failure, for example, does
    not prevent ONNX or the final report from being created.
    """
    weights_path = Path(weights_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if not weights_path.exists():
        return pd.DataFrame(
            [
                {
                    "format": fmt,
                    "status": "failed",
                    "exported_path": "",
                    "notes": "",
                    "error_message": f"Weights not found: {weights_path}",
                }
                for fmt in formats
            ]
        )

    for export_format in formats:
        export_format = str(export_format).lower()
        row: dict[str, Any] = {
            "format": export_format,
            "status": "not_started",
            "exported_path": "",
            "notes": "",
            "error_message": "",
        }
        try:
            from ultralytics import YOLO

            model = YOLO(str(weights_path))
            exported = model.export(format=export_format, imgsz=imgsz, device=device)
            exported_path = Path(str(exported))

            target_path = _target_export_path(output_dir, export_format, exported_path)
            if exported_path.exists() and exported_path.resolve() != target_path.resolve():
                shutil.copy2(exported_path, target_path)
                row["notes"] = f"copied export from {exported_path}"
            elif exported_path.exists():
                row["notes"] = "export created in final weights folder"
            else:
                target_path = exported_path
                row["notes"] = "Ultralytics returned a path that was not found on disk"

            row.update({"status": "exported", "exported_path": str(target_path)})
        except Exception as exc:
            row.update({"status": "failed", "error_message": str(exc)})
        rows.append(row)

    return pd.DataFrame(rows)


def _target_export_path(output_dir: Path, export_format: str, exported_path: Path) -> Path:
    suffix = EXPORT_EXTENSIONS.get(export_format, exported_path.suffix or f".{export_format}")
    return Path(output_dir) / f"final_model{suffix}"
