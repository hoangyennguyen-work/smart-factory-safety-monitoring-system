"""Evaluation helpers for YOLO candidate and final models."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def extract_training_metrics(run_dir: Path) -> dict[str, Any]:
    """Extract comparable metrics from an Ultralytics training run directory.

    Ultralytics has changed metric object names across releases, so this helper
    reads the stable ``results.csv`` artifact when it exists and returns ``None``
    for metrics that are unavailable.
    """
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    weights_dir = run_dir / "weights"
    best_weights_path = weights_dir / "best.pt"
    last_weights_path = weights_dir / "last.pt"

    metrics: dict[str, Any] = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fitness": None,
        "training_time_seconds": None,
        "best_weights_path": str(best_weights_path) if best_weights_path.exists() else "",
        "last_weights_path": str(last_weights_path) if last_weights_path.exists() else "",
        "model_size_mb": _file_size_mb(best_weights_path),
    }

    if not results_csv.exists():
        return metrics

    try:
        results_df = pd.read_csv(results_csv)
    except Exception:
        return metrics

    if results_df.empty:
        return metrics

    last_row = results_df.iloc[-1]
    metrics["precision"] = _row_metric(last_row, results_df.columns, ["precision"])
    metrics["recall"] = _row_metric(last_row, results_df.columns, ["recall"])
    metrics["map50"] = _row_metric(
        last_row,
        results_df.columns,
        ["map50", "mAP50"],
        exclude=["95"],
    )
    metrics["map50_95"] = _row_metric(
        last_row,
        results_df.columns,
        ["map50-95", "map50_95", "mAP50-95"],
    )
    metrics["fitness"] = _row_metric(last_row, results_df.columns, ["fitness"])
    metrics["training_time_seconds"] = _row_metric(last_row, results_df.columns, ["time"])
    return metrics


def evaluate_candidate_model(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int,
    device: str | int,
) -> dict[str, Any]:
    """Validate one trained YOLO model and return validation metrics."""
    weights_path = Path(weights_path)
    data_yaml = Path(data_yaml)
    result: dict[str, Any] = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fitness": None,
        "status": "not_run",
        "error_message": "",
    }

    if not weights_path.exists():
        result.update(
            {
                "status": "failed",
                "error_message": f"Weights not found: {weights_path}",
            }
        )
        return result

    try:
        from ultralytics import YOLO

        resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(data_yaml, weights_path.parent)
        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(resolved_data_yaml),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        result.update(_metrics_from_ultralytics_result(metrics))
        result["status"] = "evaluated"
    except Exception as exc:
        result.update({"status": "failed", "error_message": str(exc)})
    return result


def evaluate_final_model(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int,
    device: str | int,
    split: str = "test",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the locked final model on the untouched final split.

    This helper is intended for Notebook 08. It records missing metrics as
    ``None`` rather than failing when Ultralytics result attributes differ
    across versions.
    """
    weights_path = Path(weights_path)
    data_yaml = Path(data_yaml)
    result: dict[str, Any] = {
        "status": "not_run",
        "split": split,
        "weights_path": str(weights_path),
        "data_yaml": str(data_yaml),
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fitness": None,
        "validation_save_dir": "",
        "class_metrics": [],
        "error_message": "",
    }

    if not weights_path.exists():
        result.update(
            {
                "status": "failed",
                "error_message": f"Weights not found: {weights_path}",
            }
        )
        return result
    if not data_yaml.exists():
        result.update(
            {
                "status": "failed",
                "error_message": f"Dataset YAML not found: {data_yaml}",
            }
        )
        return result

    try:
        from ultralytics import YOLO

        output_dir = Path(output_dir) if output_dir is not None else weights_path.parent
        resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(data_yaml, output_dir)
        model = YOLO(str(weights_path))
        val_kwargs: dict[str, Any] = {
            "data": str(resolved_data_yaml),
            "imgsz": imgsz,
            "device": device,
            "split": split,
            "plots": True,
            "verbose": False,
        }
        if output_dir is not None:
            val_kwargs.update(
                {
                    "project": str(output_dir.parent),
                    "name": output_dir.name,
                    "exist_ok": True,
                }
            )
        metrics = model.val(**val_kwargs)
        result.update(_metrics_from_ultralytics_result(metrics))
        result["validation_save_dir"] = str(getattr(metrics, "save_dir", ""))
        result["class_metrics"] = _class_metrics_from_ultralytics_result(metrics)
        result["status"] = "evaluated"
    except Exception as exc:
        result.update({"status": "failed", "error_message": str(exc)})
    return result


def save_final_predictions(
    weights_path: Path,
    source_dir: Path,
    output_dir: Path,
    imgsz: int,
    device: str | int,
    conf: float = 0.25,
) -> dict[str, Any]:
    """Save visual predictions for qualitative final-test inspection."""
    weights_path = Path(weights_path)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    result: dict[str, Any] = {
        "status": "not_run",
        "weights_path": str(weights_path),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "num_source_images": 0,
        "error_message": "",
    }

    if not weights_path.exists():
        result.update(
            {
                "status": "failed",
                "error_message": f"Weights not found: {weights_path}",
            }
        )
        return result
    image_paths = _collect_sample_images(source_dir, max_images=10_000)
    if not image_paths:
        result.update(
            {
                "status": "failed",
                "error_message": f"No source images found in {source_dir}",
            }
        )
        return result

    try:
        from ultralytics import YOLO

        output_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(weights_path))
        model.predict(
            source=str(source_dir),
            imgsz=imgsz,
            device=device,
            conf=conf,
            save=True,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            verbose=False,
        )
        result.update(
            {
                "status": "saved",
                "num_source_images": len(image_paths),
            }
        )
    except Exception as exc:
        result.update({"status": "failed", "error_message": str(exc)})
    return result


def benchmark_model_latency(
    weights_path: Path,
    sample_images_dir: Path,
    imgsz: int,
    device: str | int,
    max_images: int = 20,
) -> dict[str, Any]:
    """Benchmark prediction latency on a small validation image subset."""
    weights_path = Path(weights_path)
    sample_images_dir = Path(sample_images_dir)
    result: dict[str, Any] = {
        "latency_status": "not_run",
        "benchmark_images": 0,
        "avg_inference_ms": None,
        "fps": None,
        "latency_error": "",
    }

    if not weights_path.exists():
        result.update(
            {
                "latency_status": "failed",
                "latency_error": f"Weights not found: {weights_path}",
            }
        )
        return result

    image_paths = _collect_sample_images(sample_images_dir, max_images=max_images)
    if not image_paths:
        result.update(
            {
                "latency_status": "failed",
                "latency_error": f"No sample images found in {sample_images_dir}",
            }
        )
        return result

    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        # Warm up once so first-call setup overhead does not dominate timing.
        model.predict(
            source=str(image_paths[0]),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        timings: list[float] = []
        for image_path in image_paths:
            start_time = time.perf_counter()
            model.predict(
                source=str(image_path),
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            timings.append(time.perf_counter() - start_time)

        avg_seconds = sum(timings) / len(timings)
        result.update(
            {
                "latency_status": "benchmarked",
                "benchmark_images": len(timings),
                "avg_inference_ms": avg_seconds * 1000,
                "fps": 1.0 / avg_seconds if avg_seconds > 0 else None,
            }
        )
    except Exception as exc:
        result.update({"latency_status": "failed", "latency_error": str(exc)})
    return result


def rank_candidate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by validation recall, mAP, FPS, then model size."""
    if results_df.empty:
        return results_df.copy()

    ranked = results_df.copy()
    for column in ["recall", "map50", "map50_95", "fps"]:
        if column not in ranked:
            ranked[column] = None
    if "model_size_mb" not in ranked:
        ranked["model_size_mb"] = None

    successful = ranked["status"].eq("trained") if "status" in ranked else True
    ranked["rank_score"] = (
        ranked["recall"].fillna(-1) * 1_000_000_000
        + ranked["map50"].fillna(-1) * 1_000_000
        + ranked["map50_95"].fillna(-1) * 1_000
        + ranked["fps"].fillna(-1)
        - ranked["model_size_mb"].fillna(1_000_000) / 1_000_000
    )
    ranked.loc[~successful, "rank_score"] = -1
    return ranked.sort_values(
        by=["rank_score", "recall", "map50", "map50_95", "fps", "model_size_mb"],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def rank_ablation_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rank ablation runs by validation recall, mAP, then inference speed."""
    if results_df.empty:
        return results_df.copy()

    ranked = results_df.copy()
    for column in ["recall", "map50", "map50_95", "fps"]:
        if column not in ranked:
            ranked[column] = None

    successful = ranked["status"].eq("trained") if "status" in ranked else True
    ranked["rank_score"] = (
        ranked["recall"].fillna(-1) * 1_000_000_000
        + ranked["map50"].fillna(-1) * 1_000_000
        + ranked["map50_95"].fillna(-1) * 1_000
        + ranked["fps"].fillna(-1)
    )
    ranked.loc[~successful, "rank_score"] = -1
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked = ranked.sort_values(
        by=["rank_score", "recall", "map50", "map50_95", "fps"],
        ascending=[False, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def evaluate_yolo_model(model_path: Path, data_config: Path) -> dict[str, Any]:
    """Backward-compatible validation wrapper."""
    return evaluate_candidate_model(
        weights_path=model_path,
        data_yaml=data_config,
        imgsz=640,
        device=0,
    )


def _metrics_from_ultralytics_result(metrics: Any) -> dict[str, Any]:
    result = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fitness": None,
    }
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        result["precision"] = _dict_metric(results_dict, ["precision"])
        result["recall"] = _dict_metric(results_dict, ["recall"])
        result["map50"] = _dict_metric(results_dict, ["map50"], exclude=["95"])
        result["map50_95"] = _dict_metric(results_dict, ["map50-95", "map50_95"])
        result["fitness"] = _dict_metric(results_dict, ["fitness"])

    box = getattr(metrics, "box", None)
    if box is not None:
        result["precision"] = result["precision"] if result["precision"] is not None else getattr(box, "mp", None)
        result["recall"] = result["recall"] if result["recall"] is not None else getattr(box, "mr", None)
        result["map50"] = result["map50"] if result["map50"] is not None else getattr(box, "map50", None)
        result["map50_95"] = result["map50_95"] if result["map50_95"] is not None else getattr(box, "map", None)
    return result


def _class_metrics_from_ultralytics_result(metrics: Any) -> list[dict[str, Any]]:
    names = getattr(metrics, "names", {}) or {}
    box = getattr(metrics, "box", None)
    if box is None:
        return []

    rows: list[dict[str, Any]] = []
    for class_id, class_name in sorted(names.items()):
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }
        try:
            class_result = box.class_result(int(class_id))
            if len(class_result) >= 4:
                row.update(
                    {
                        "precision": _safe_float(class_result[0]),
                        "recall": _safe_float(class_result[1]),
                        "map50": _safe_float(class_result[2]),
                        "map50_95": _safe_float(class_result[3]),
                    }
                )
        except Exception:
            row.update(
                {
                    "precision": _sequence_value(getattr(box, "p", None), int(class_id)),
                    "recall": _sequence_value(getattr(box, "r", None), int(class_id)),
                    "map50": _sequence_value(getattr(box, "ap50", None), int(class_id)),
                    "map50_95": _sequence_value(getattr(box, "maps", None), int(class_id)),
                }
            )
        rows.append(row)
    return rows


def _row_metric(
    row: pd.Series,
    columns: pd.Index,
    candidates: list[str],
    exclude: list[str] | None = None,
) -> float | None:
    column = _find_column(columns, candidates, exclude=exclude)
    if column is None:
        return None
    value = row[column]
    return None if pd.isna(value) else float(value)


def _dict_metric(
    values: dict[str, Any],
    candidates: list[str],
    exclude: list[str] | None = None,
) -> float | None:
    column = _find_column(pd.Index(values.keys()), candidates, exclude=exclude)
    if column is None:
        return None
    value = values[column]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sequence_value(values: Any, index: int) -> float | None:
    try:
        return _safe_float(values[index])
    except Exception:
        return None


def _find_column(
    columns: pd.Index,
    candidates: list[str],
    exclude: list[str] | None = None,
) -> str | None:
    exclude = [item.lower() for item in (exclude or [])]
    for column in columns:
        normalized = str(column).lower().replace("_", "").replace("/", "")
        if any(excluded in normalized for excluded in exclude):
            continue
        for candidate in candidates:
            normalized_candidate = candidate.lower().replace("_", "").replace("/", "")
            if normalized_candidate in normalized:
                return str(column)
    return None


def _collect_sample_images(sample_images_dir: Path, max_images: int) -> list[Path]:
    if not Path(sample_images_dir).exists():
        return []
    image_paths = [
        path
        for path in sorted(Path(sample_images_dir).iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    return image_paths[:max_images]


def _file_size_mb(path: Path) -> float | None:
    if not Path(path).exists():
        return None
    return Path(path).stat().st_size / (1024 * 1024)


def _resolve_dataset_yaml_for_ultralytics(data_yaml: Path, output_dir: Path) -> Path:
    """Write a runtime YAML with an absolute dataset root when needed."""
    data_yaml = Path(data_yaml).resolve()
    with data_yaml.open("r", encoding="utf-8") as file_handle:
        data_config = yaml.safe_load(file_handle)

    dataset_path = Path(str(data_config.get("path", "")))
    if dataset_path.is_absolute():
        return data_yaml

    resolved_config = dict(data_config)
    resolved_config["path"] = str((data_yaml.parent / dataset_path).resolve())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = output_dir / f"_{data_yaml.stem}_resolved.yaml"
    _write_runtime_yaml(resolved_yaml, resolved_config)
    return resolved_yaml


def _write_runtime_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Refresh a generated runtime YAML, replacing only this helper file."""
    path = Path(path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)
    temporary_path.replace(path)
