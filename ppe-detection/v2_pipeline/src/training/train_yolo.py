"""YOLO training entry points used by the v2 notebooks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.training.evaluate_yolo import extract_training_metrics


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

MODEL_RUN_NAMES = {
    "yolov8n.pt": "YOLOv8_Nano_expD",
    "yolov9t.pt": "YOLOv9_Tiny_expD",
    "yolov10n.pt": "YOLOv10_Nano_expD",
    "yolo11n.pt": "YOLO11_Nano_expD",
    "yolo12n.pt": "YOLO12_Nano_expD",
    "yolo26n.pt": "YOLO26_Nano_expD",
}

ABLATION_EXPERIMENT_RUN_NAMES = {
    "exp_A_original_only": "A_original_only",
    "exp_B_online_aug": "B_online_aug",
    "exp_C_offline_aug": "C_offline_aug",
    "exp_D_full_pipeline": "D_full_pipeline",
}

ONLINE_AUGMENTATION_EXPERIMENTS = {"exp_B_online_aug", "exp_D_full_pipeline"}


def train_candidate_models(
    data_yaml: Path,
    model_names: list[str],
    output_dir: Path,
    train_args: dict[str, Any],
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Train YOLO candidate models with shared settings.

    Args:
        data_yaml: Ultralytics dataset YAML. Notebook 06 uses
            ``data_exp_D_full_pipeline.yaml`` for architecture triage.
        model_names: Candidate checkpoint names, such as ``yolov8n.pt``.
        output_dir: Parent run directory for candidate model runs.
        train_args: Shared Ultralytics training arguments. Custom keys supported
            here are ``dry_run`` and ``overwrite``.
        continue_on_error: If ``True``, failed candidates are recorded and the
            next model is attempted.

    Returns:
        One row per candidate with status, run path, metrics, and errors.
    """
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dry_run = bool(train_args.get("dry_run", False))
    overwrite = bool(train_args.get("overwrite", False))
    rows: list[dict[str, Any]] = []

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(data_yaml, output_dir)

    for model_name in model_names:
        base_run_name = _run_name_for_model(model_name)
        run_name = (
            base_run_name
            if overwrite
            else _unique_run_name(output_dir=output_dir, base_name=base_run_name)
        )
        run_dir = output_dir / run_name
        row = _base_report_row(model_name=model_name, run_name=run_name, run_dir=run_dir)

        try:
            if dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                row.update(
                    {
                        "status": "dry_run",
                        "error_message": "",
                        "notes": "training skipped because dry_run=True",
                    }
                )
            else:
                train_result = _train_one_candidate(
                    model_name=model_name,
                    data_yaml=resolved_data_yaml,
                    output_dir=output_dir,
                    run_name=run_name,
                    train_args=train_args,
                )
                actual_run_dir = _resolve_run_dir(train_result, fallback=run_dir)
                metrics = extract_training_metrics(actual_run_dir)
                row.update(
                    {
                        "status": "trained",
                        "run_dir": str(actual_run_dir),
                        "error_message": "",
                        **metrics,
                    }
                )
        except Exception as exc:
            row.update({"status": "failed", "error_message": str(exc)})
            if not continue_on_error:
                rows.append(row)
                raise

        rows.append(row)

    return pd.DataFrame(rows)


def train_ablation_experiments(
    experiment_data_yamls: dict[str, Path],
    model_name: str,
    output_dir: Path,
    base_train_args: dict[str, Any],
    online_augmentation_args: dict[str, Any] | None = None,
    online_aug_experiments: set[str] | None = None,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Train ablation experiments with one fixed YOLO architecture.

    Notebook 07 should call this only after Notebook 06 has selected an
    architecture. Experiments A/C receive offline/no online augmentation
    settings, while B/D receive the online augmentation arguments.

    Args:
        experiment_data_yamls: Mapping of experiment name to dataset YAML.
        model_name: Selected YOLO checkpoint or model YAML.
        output_dir: Parent run directory for ablation runs.
        base_train_args: Shared Ultralytics training arguments. Custom keys
            supported here are ``dry_run`` and ``overwrite``.
        online_augmentation_args: Ultralytics online augmentation arguments for
            experiments B and D.
        online_aug_experiments: Experiment names that should use online
            augmentation. Defaults to B and D.
        continue_on_error: If ``True``, failed experiments are recorded and the
            next experiment is attempted.

    Returns:
        One row per ablation experiment with status, run path, metrics, and
        errors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not experiment_data_yamls:
        raise ValueError("No experiment dataset YAML files were provided.")

    dry_run = bool(base_train_args.get("dry_run", False))
    overwrite = bool(base_train_args.get("overwrite", False))
    online_aug_experiments = online_aug_experiments or ONLINE_AUGMENTATION_EXPERIMENTS
    online_augmentation_args = online_augmentation_args or {}
    rows: list[dict[str, Any]] = []

    for experiment_name in sorted(experiment_data_yamls):
        data_yaml = Path(experiment_data_yamls[experiment_name])
        run_base_name = _run_name_for_ablation(
            model_name=model_name,
            experiment_name=experiment_name,
        )
        run_name = (
            run_base_name
            if overwrite
            else _unique_run_name(output_dir=output_dir, base_name=run_base_name)
        )
        run_dir = output_dir / run_name
        uses_online_aug = experiment_name in online_aug_experiments
        row = _base_ablation_report_row(
            experiment_name=experiment_name,
            model_name=model_name,
            run_name=run_name,
            run_dir=run_dir,
            data_yaml=data_yaml,
            uses_online_aug=uses_online_aug,
        )

        try:
            if not data_yaml.exists():
                raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

            train_args = _build_ablation_train_args(
                base_train_args=base_train_args,
                online_augmentation_args=online_augmentation_args,
                use_online_augmentation=uses_online_aug,
            )
            resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(
                data_yaml=data_yaml,
                output_dir=output_dir,
            )

            if dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                row.update(
                    {
                        "status": "dry_run",
                        "resolved_data_yaml": str(resolved_data_yaml),
                        "error_message": "",
                        "notes": "training skipped because dry_run=True",
                    }
                )
            else:
                train_result = _train_one_candidate(
                    model_name=model_name,
                    data_yaml=resolved_data_yaml,
                    output_dir=output_dir,
                    run_name=run_name,
                    train_args=train_args,
                )
                actual_run_dir = _resolve_run_dir(train_result, fallback=run_dir)
                metrics = extract_training_metrics(actual_run_dir)
                row.update(
                    {
                        "status": "trained",
                        "run_dir": str(actual_run_dir),
                        "resolved_data_yaml": str(resolved_data_yaml),
                        "error_message": "",
                        **metrics,
                    }
                )
        except Exception as exc:
            row.update({"status": "failed", "error_message": str(exc)})
            if not continue_on_error:
                rows.append(row)
                raise

        rows.append(row)

    return pd.DataFrame(rows)


def train_final_model(
    selected_model: str,
    data_yaml: Path,
    output_dir: Path,
    run_name: str,
    train_args: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train one final YOLO model after architecture and config selection.

    The final training step should be called only after candidate triage and
    ablation choices are locked. Custom keys supported in ``train_args`` are
    ``dry_run`` and ``selected_experiment``.
    """
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    selected_experiment = str(train_args.get("selected_experiment", ""))
    base_run_name = run_name or _run_name_for_final(selected_model, selected_experiment)
    safe_run_name = (
        base_run_name
        if overwrite
        else _unique_run_name(output_dir=output_dir, base_name=base_run_name)
    )
    run_dir = output_dir / safe_run_name
    resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(data_yaml, output_dir)

    result = {
        "selected_model": selected_model,
        "selected_experiment": selected_experiment,
        "data_yaml": str(data_yaml),
        "resolved_data_yaml": str(resolved_data_yaml),
        "run_name": safe_run_name,
        "run_dir": str(run_dir),
        "best_weights_path": "",
        "last_weights_path": "",
        "training_status": "not_started",
        "notes": "",
        "error_message": "",
    }

    try:
        if bool(train_args.get("dry_run", False)):
            run_dir.mkdir(parents=True, exist_ok=True)
            result.update(
                {
                    "training_status": "dry_run",
                    "notes": "training skipped because dry_run=True",
                }
            )
            return result

        effective_args = dict(train_args)
        effective_args["overwrite"] = overwrite
        train_result = _train_one_candidate(
            model_name=selected_model,
            data_yaml=resolved_data_yaml,
            output_dir=output_dir,
            run_name=safe_run_name,
            train_args=effective_args,
        )
        actual_run_dir = _resolve_run_dir(train_result, fallback=run_dir)
        metrics = extract_training_metrics(actual_run_dir)
        result.update(
            {
                "training_status": "trained",
                "run_dir": str(actual_run_dir),
                "best_weights_path": metrics.get("best_weights_path", ""),
                "last_weights_path": metrics.get("last_weights_path", ""),
                "notes": "final training completed",
                **metrics,
            }
        )
    except Exception as exc:
        result.update(
            {
                "training_status": "failed",
                "error_message": str(exc),
                "notes": "final training failed",
            }
        )
    return result


def train_yolo_model(
    data_config: Path,
    model_name: str,
    run_dir: Path,
    epochs: int = 100,
) -> dict[str, str]:
    """Backward-compatible wrapper for training a single YOLO model."""
    run_dir = Path(run_dir)
    results = train_candidate_models(
        data_yaml=data_config,
        model_names=[model_name],
        output_dir=run_dir.parent,
        train_args={"epochs": epochs, "dry_run": False, "overwrite": False},
    )
    return results.iloc[0].to_dict()


def _train_one_candidate(
    model_name: str,
    data_yaml: Path,
    output_dir: Path,
    run_name: str,
    train_args: dict[str, Any],
) -> Any:
    from ultralytics import YOLO

    custom_keys = {"dry_run", "overwrite", "selected_experiment"}
    yolo_args = {
        key: value
        for key, value in train_args.items()
        if key not in custom_keys and value is not None
    }
    yolo_args.update(
        {
            "data": str(data_yaml),
            "project": str(output_dir),
            "name": run_name,
            "exist_ok": bool(train_args.get("overwrite", False)),
        }
    )

    model = YOLO(model_name)
    return model.train(**yolo_args)


def _resolve_dataset_yaml_for_ultralytics(data_yaml: Path, output_dir: Path) -> Path:
    """Write a training-time YAML whose dataset root is absolute.

    Ultralytics resolves relative ``path`` entries against its global datasets
    directory in some versions. The repo keeps dataset YAML files relative to
    ``v2_pipeline`` for portability, so this helper creates an ignored runtime
    copy with an absolute ``path`` before training.
    """
    data_yaml = Path(data_yaml).resolve()
    with data_yaml.open("r", encoding="utf-8") as file_handle:
        data_config = yaml.safe_load(file_handle)

    dataset_path = Path(str(data_config.get("path", "")))
    if dataset_path.is_absolute():
        return data_yaml

    resolved_dataset_path = (data_yaml.parent / dataset_path).resolve()
    resolved_config = dict(data_config)
    resolved_config["path"] = str(resolved_dataset_path)

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


def _run_name_for_model(model_name: str) -> str:
    normalized = Path(model_name).name
    return MODEL_RUN_NAMES.get(normalized, f"{Path(normalized).stem}_expD")


def _run_name_for_ablation(model_name: str, experiment_name: str) -> str:
    model_stem = Path(model_name).stem.replace(".", "_")
    experiment_label = ABLATION_EXPERIMENT_RUN_NAMES.get(
        experiment_name,
        experiment_name,
    )
    return f"{model_stem}_{experiment_label}"


def _run_name_for_final(model_name: str, experiment_name: str) -> str:
    model_stem = Path(model_name).stem.replace(".", "_")
    experiment_label = ABLATION_EXPERIMENT_RUN_NAMES.get(
        experiment_name,
        experiment_name or "selected_config",
    )
    return f"final_{model_stem}_{experiment_label}"


def _build_ablation_train_args(
    base_train_args: dict[str, Any],
    online_augmentation_args: dict[str, Any],
    use_online_augmentation: bool,
) -> dict[str, Any]:
    train_args = dict(base_train_args)
    if use_online_augmentation:
        train_args.update(online_augmentation_args)
    else:
        train_args.update(_online_augmentation_off_args(online_augmentation_args))
    return train_args


def _online_augmentation_off_args(
    online_augmentation_args: dict[str, Any],
) -> dict[str, Any]:
    """Return neutral values for the online augmentation keys in config."""
    off_values: dict[str, Any] = {}
    for key in online_augmentation_args:
        if key == "close_mosaic":
            off_values[key] = 0
        else:
            off_values[key] = 0.0
    return off_values


def _unique_run_name(output_dir: Path, base_name: str) -> str:
    candidate = base_name
    suffix = 2
    while (Path(output_dir) / candidate).exists():
        candidate = f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _resolve_run_dir(train_result: Any, fallback: Path) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir is not None:
        return Path(save_dir)

    trainer = getattr(train_result, "trainer", None)
    if trainer is not None and getattr(trainer, "save_dir", None) is not None:
        return Path(trainer.save_dir)

    return Path(fallback)


def _base_report_row(model_name: str, run_name: str, run_dir: Path) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "run_name": run_name,
        "status": "not_started",
        "run_dir": str(run_dir),
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fitness": None,
        "training_time_seconds": None,
        "best_weights_path": "",
        "last_weights_path": "",
        "model_size_mb": None,
        "error_message": "",
        "notes": "",
    }


def _base_ablation_report_row(
    experiment_name: str,
    model_name: str,
    run_name: str,
    run_dir: Path,
    data_yaml: Path,
    uses_online_aug: bool,
) -> dict[str, Any]:
    row = _base_report_row(model_name=model_name, run_name=run_name, run_dir=run_dir)
    row.update(
        {
            "experiment": experiment_name,
            "data_yaml": str(data_yaml),
            "resolved_data_yaml": "",
            "uses_online_augmentation": uses_online_aug,
        }
    )
    return row
