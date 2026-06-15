"""YOLO evaluation utilities for factory sign detection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _last_metric(results_df: pd.DataFrame, candidates: list[str]) -> float | None:
    for column in candidates:
        if column in results_df.columns:
            series = results_df[column].dropna()
            if not series.empty:
                return _safe_float(series.iloc[-1])
    return None


def extract_training_metrics(run_dir: Path) -> dict:
    """Extract validation metrics and weight paths from a completed Ultralytics run."""
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    weights_dir = run_dir / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    metrics: dict[str, object] = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fitness": None,
        "training_time": None,
        "best_weights_path": str(best_weights) if best_weights.exists() else "",
        "last_weights_path": str(last_weights) if last_weights.exists() else "",
        "model_size_mb": best_weights.stat().st_size / (1024 * 1024) if best_weights.exists() else None,
    }
    if not results_csv.exists():
        return metrics

    try:
        results_df = pd.read_csv(results_csv)
    except Exception:
        return metrics

    # Ultralytics metric column names vary across versions, so try common variants.
    metrics["precision"] = _last_metric(results_df, ["metrics/precision(B)", "precision", "metrics/precision"])
    metrics["recall"] = _last_metric(results_df, ["metrics/recall(B)", "recall", "metrics/recall"])
    metrics["map50"] = _last_metric(results_df, ["metrics/mAP50(B)", "mAP50", "map50", "metrics/mAP50"])
    metrics["map50_95"] = _last_metric(
        results_df,
        ["metrics/mAP50-95(B)", "mAP50-95", "map50_95", "metrics/mAP50-95"],
    )
    metrics["fitness"] = _last_metric(results_df, ["fitness"])
    metrics["training_time"] = _last_metric(results_df, ["time", "train/time", "epoch_time"])
    return metrics


def _metrics_from_ultralytics_result(result: Any) -> dict[str, float | None]:
    box = getattr(result, "box", None)
    if box is None:
        return {"precision": None, "recall": None, "map50": None, "map50_95": None, "fitness": None}
    fitness_attr = getattr(result, "fitness", None)
    fitness = fitness_attr() if callable(fitness_attr) else fitness_attr
    return {
        "precision": _safe_float(getattr(box, "mp", None)),
        "recall": _safe_float(getattr(box, "mr", None)),
        "map50": _safe_float(getattr(box, "map50", None)),
        "map50_95": _safe_float(getattr(box, "map", None)),
        "fitness": _safe_float(fitness),
    }


def evaluate_candidate_model(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int,
    device: str | int,
) -> dict:
    """Run validation on a trained candidate model and return best-effort metrics."""
    try:
        from ultralytics import YOLO

        resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(data_yaml, weights_path.parent)
        model = YOLO(str(weights_path))
        # Candidate triage uses validation only. The test split is reserved for final evaluation.
        result = model.val(data=str(resolved_data_yaml), imgsz=imgsz, device=device, split="val", verbose=False)
        metrics = _metrics_from_ultralytics_result(result)
        metrics.update({"status": "evaluated", "error_message": ""})
        return metrics
    except Exception as exc:
        return {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "fitness": None,
            "status": "failed",
            "error_message": str(exc),
        }


def evaluate_final_model(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int,
    device: str | int,
    split: str = "test",
    output_dir: Path | None = None,
    plots: bool = True,
) -> dict:
    """Evaluate the locked final model on the untouched test split.

    This helper is intended for Notebook 07 only. It runs Ultralytics validation
    on ``split='test'`` by default and records unavailable metrics as ``None``
    because Ultralytics result attributes vary by version.
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
        result.update({"status": "failed", "error_message": f"Weights not found: {weights_path}"})
        return result
    if not data_yaml.exists():
        result.update({"status": "failed", "error_message": f"Dataset YAML not found: {data_yaml}"})
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
            "plots": plots,
            "verbose": False,
        }
        if output_dir is not None:
            val_kwargs.update({"project": str(output_dir.parent), "name": output_dir.name, "exist_ok": True})

        # The test split is used here for final reporting only, never for tuning.
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
) -> dict:
    """Save optional final-test prediction images for qualitative review."""
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
        result.update({"status": "failed", "error_message": f"Weights not found: {weights_path}"})
        return result
    image_paths = _collect_sample_images(source_dir, max_images=10_000)
    if not image_paths:
        result.update({"status": "failed", "error_message": f"No source images found in {source_dir}"})
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
        result.update({"status": "saved", "num_source_images": len(image_paths)})
    except Exception as exc:
        result.update({"status": "failed", "error_message": str(exc)})
    return result


def benchmark_model_latency(
    weights_path: Path,
    sample_images_dir: Path,
    imgsz: int,
    device: str | int,
    max_images: int = 20,
) -> dict:
    """Estimate average prediction latency on a deterministic validation image subset."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    sample_images = [
        path
        for path in sorted(Path(sample_images_dir).iterdir()) if path.is_file() and path.suffix.lower() in image_extensions
    ][:max_images]
    if not sample_images:
        return {
            "avg_latency_ms": None,
            "fps": None,
            "num_benchmark_images": 0,
            "latency_warning": "no validation images available for benchmarking",
        }

    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        # One warm-up pass reduces first-call overhead from model setup.
        model.predict(source=str(sample_images[0]), imgsz=imgsz, device=device, verbose=False)
        start = time.perf_counter()
        for image_path in sample_images:
            model.predict(source=str(image_path), imgsz=imgsz, device=device, verbose=False)
        elapsed = time.perf_counter() - start
        avg_ms = elapsed / len(sample_images) * 1000
        return {
            "avg_latency_ms": avg_ms,
            "fps": 1000 / avg_ms if avg_ms > 0 else None,
            "num_benchmark_images": len(sample_images),
            "latency_warning": "",
        }
    except Exception as exc:
        return {
            "avg_latency_ms": None,
            "fps": None,
            "num_benchmark_images": len(sample_images),
            "latency_warning": str(exc),
        }


def rank_candidate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rank successful candidates by validation recall, mAP, FPS, then model size."""
    if results_df.empty:
        return results_df.copy()
    ranked = results_df.copy()
    successful = ranked["status"].eq("trained") if "status" in ranked.columns else pd.Series(False, index=ranked.index)
    ranked = ranked.loc[successful].copy()
    if ranked.empty:
        ranked["rank"] = pd.Series(dtype="int")
        return ranked

    for column in ["recall", "map50", "map50_95", "fps", "model_size_mb"]:
        if column not in ranked.columns:
            ranked[column] = None
    ranked = ranked.sort_values(
        by=["recall", "map50", "map50_95", "fps", "model_size_mb"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def rank_ablation_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rank successful ablation runs by validation metrics only."""
    if results_df.empty:
        return results_df.copy()
    ranked = results_df.copy()
    successful = ranked["status"].eq("trained") if "status" in ranked.columns else pd.Series(False, index=ranked.index)
    ranked = ranked.loc[successful].copy()
    if ranked.empty:
        ranked["rank"] = pd.Series(dtype="int")
        return ranked

    for column in ["recall", "map50", "map50_95", "fps", "model_size_mb"]:
        if column not in ranked.columns:
            ranked[column] = None
    ranked = ranked.sort_values(
        by=["recall", "map50", "map50_95", "fps", "model_size_mb"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


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


def _sequence_value(values: Any, index: int) -> float | None:
    try:
        return _safe_float(values[index])
    except Exception:
        return None


def _collect_sample_images(sample_images_dir: Path, max_images: int) -> list[Path]:
    image_extensions = {".jpg", ".jpeg", ".png"}
    if not Path(sample_images_dir).exists():
        return []
    return [
        path
        for path in sorted(Path(sample_images_dir).iterdir())
        if path.is_file() and path.suffix.lower() in image_extensions
    ][:max_images]


def _resolve_dataset_yaml_for_ultralytics(data_yaml: Path, output_dir: Path) -> Path:
    """Write a runtime YAML with an absolute dataset root when needed."""
    data_yaml = Path(data_yaml).resolve()
    with data_yaml.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    dataset_path = Path(str(config.get("path", "")))
    if dataset_path.is_absolute():
        return data_yaml

    resolved_config = dict(config)
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
