"""Model export helpers for downstream deployment targets."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd


EXPORT_SUFFIXES = {
    "onnx": ".onnx",
    "engine": ".engine",
    "torchscript": ".torchscript",
    "openvino": "_openvino_model",
}


def export_final_model(
    weights_path: Path,
    output_dir: Path,
    formats: list[str],
    imgsz: int,
    device: str | int,
) -> pd.DataFrame:
    """Export final YOLO weights to deployment formats.

    Each requested format is attempted independently. Failures are recorded in
    the returned report so TensorRT or optional exporter issues do not block
    ONNX or the rest of Notebook 08.
    """
    weights_path = Path(weights_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if not weights_path.exists():
        return pd.DataFrame(
            [
                {
                    "format": export_format,
                    "status": "failed",
                    "source_weights": str(weights_path),
                    "exported_path": "",
                    "copied_path": "",
                    "error_message": f"Weights not found: {weights_path}",
                }
                for export_format in formats
            ]
        )

    for export_format in formats:
        row = {
            "format": export_format,
            "status": "not_started",
            "source_weights": str(weights_path),
            "exported_path": "",
            "copied_path": "",
            "error_message": "",
        }
        try:
            from ultralytics import YOLO

            model = YOLO(str(weights_path))
            exported = model.export(
                format=export_format,
                imgsz=imgsz,
                device=device,
            )
            exported_path = _resolve_export_path(exported)
            copied_path = _copy_export_artifact(
                exported_path=exported_path,
                output_dir=output_dir,
                export_format=export_format,
            )
            row.update(
                {
                    "status": "exported",
                    "exported_path": str(exported_path),
                    "copied_path": str(copied_path),
                }
            )
        except Exception as exc:
            row.update({"status": "failed", "error_message": str(exc)})
        rows.append(row)

    return pd.DataFrame(rows)


def export_model(model_path: Path, export_format: str = "onnx") -> Path:
    """Export a final YOLO model to a deployment-friendly format."""
    model_path = Path(model_path)
    report = export_final_model(
        weights_path=model_path,
        output_dir=model_path.parent,
        formats=[export_format],
        imgsz=640,
        device=0,
    )
    copied_path = report.iloc[0].get("copied_path", "")
    return Path(copied_path) if copied_path else model_path


def _resolve_export_path(exported: Any) -> Path:
    if isinstance(exported, (str, Path)):
        return Path(exported)
    if isinstance(exported, list) and exported:
        return Path(exported[0])
    raise ValueError(f"Could not resolve export artifact path from: {exported!r}")


def _copy_export_artifact(
    exported_path: Path,
    output_dir: Path,
    export_format: str,
) -> Path:
    exported_path = Path(exported_path)
    output_dir = Path(output_dir)
    if not exported_path.exists():
        raise FileNotFoundError(f"Export artifact not found: {exported_path}")

    if exported_path.is_dir():
        target_dir = output_dir / exported_path.name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(exported_path, target_dir)
        return target_dir

    suffix = EXPORT_SUFFIXES.get(export_format, exported_path.suffix)
    target_path = output_dir / f"final_model{suffix}"
    shutil.copy2(exported_path, target_path)
    return target_path
