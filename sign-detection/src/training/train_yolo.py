"""YOLO training utilities for factory sign detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.training.evaluate_yolo import extract_training_metrics


# These defaults avoid common Windows/Jupyter conflicts between torch, OpenMP,
# and MKL when Ultralytics starts dataloaders from a notebook kernel.
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

ABLATION_RUN_PREFIXES = {
    "exp_A_original_only": "A_original_only",
    "exp_B_online_aug": "B_online_aug",
    "exp_C_offline_aug": "C_offline_aug",
    "exp_D_full_pipeline": "D_full_pipeline",
}


def train_candidate_models(
    data_yaml: Path,
    candidate_models: list[dict],
    output_dir: Path,
    train_args: dict,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Train lightweight YOLO candidates on the same full-pipeline dataset.

    Notebook 05 uses ``data_exp_D_full_pipeline.yaml`` for every candidate so
    architecture comparison is fair: the model is the only thing changing.
    Training failures are captured as report rows, which lets the notebook skip
    unsupported checkpoints without losing the remaining candidate runs.

    Args:
        data_yaml: Ultralytics dataset YAML for ``exp_D_full_pipeline``.
        candidate_models: Candidate configs with ``name`` and ``weights`` keys.
        output_dir: Parent directory for candidate training runs.
        train_args: Shared Ultralytics training arguments. Custom keys supported
            by this helper are ``dry_run`` and ``overwrite``.
        continue_on_error: Continue after a failed candidate when ``True``.

    Returns:
        A report DataFrame containing one row per candidate model.
    """
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    dry_run = bool(train_args.get("dry_run", False))
    overwrite = bool(train_args.get("overwrite", False))
    resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(data_yaml, output_dir)

    rows: list[dict[str, Any]] = []
    for candidate in candidate_models:
        model_name = _candidate_name(candidate)
        weights = _candidate_weights(candidate)
        base_run_name = _candidate_run_name(candidate)
        run_name = base_run_name if overwrite else _unique_run_name(output_dir, base_run_name)
        run_dir = output_dir / run_name
        row = _base_candidate_row(
            model_name=model_name,
            weights=weights,
            run_name=run_name,
            run_dir=run_dir,
            data_yaml=data_yaml,
            resolved_data_yaml=resolved_data_yaml,
        )

        try:
            if dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                row.update(
                    {
                        "status": "dry_run",
                        "notes": "training skipped because dry_run=True",
                    }
                )
            else:
                train_result = _train_one_candidate(
                    weights=weights,
                    data_yaml=resolved_data_yaml,
                    output_dir=output_dir,
                    run_name=run_name,
                    train_args=train_args,
                )
                actual_run_dir = _resolve_run_dir(train_result, fallback=run_dir)
                row.update(
                    {
                        "status": "trained",
                        "run_dir": str(actual_run_dir),
                        "error_message": "",
                        **extract_training_metrics(actual_run_dir),
                    }
                )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error_message": str(exc),
                    "notes": "candidate failed; remaining candidates can continue",
                }
            )
            rows.append(row)
            if not continue_on_error:
                raise
            continue

        rows.append(row)

    return pd.DataFrame(rows)


def train_ablation_experiments(
    selected_model: str,
    experiment_configs: list[dict],
    output_dir: Path,
    base_train_args: dict,
    online_aug_args: dict,
    offline_aug_args: dict,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Train one selected architecture across all ablation configurations.

    Notebook 06 compares training configurations, not architectures. The same
    selected checkpoint from Notebook 05 is reused for every experiment, while
    dataset YAML and online augmentation settings change according to A/B/C/D.
    Failures are recorded row-by-row so one broken run does not hide the rest
    of the ablation study.

    Args:
        selected_model: Fixed YOLO checkpoint selected by candidate triage.
        experiment_configs: Experiment dictionaries with ``experiment``,
            ``dataset_yaml``, and ``use_online_augmentation`` fields.
        output_dir: Parent directory for ablation runs.
        base_train_args: Shared Ultralytics training arguments. Custom keys
            supported by this helper are ``dry_run`` and ``overwrite``.
        online_aug_args: Ultralytics augmentation settings for B and D.
        offline_aug_args: Neutral/minimal augmentation settings for A and C.
        continue_on_error: Continue after a failed experiment when ``True``.

    Returns:
        A report DataFrame containing one row per ablation experiment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not selected_model:
        raise ValueError("selected_model is required for ablation training.")
    if not experiment_configs:
        raise ValueError("No ablation experiment configs were provided.")

    dry_run = bool(base_train_args.get("dry_run", False))
    overwrite = bool(base_train_args.get("overwrite", False))
    rows: list[dict[str, Any]] = []

    for experiment_config in experiment_configs:
        experiment = str(experiment_config.get("experiment", ""))
        dataset_yaml = Path(str(experiment_config.get("dataset_yaml", "")))
        use_online_aug = bool(experiment_config.get("use_online_augmentation", False))
        run_base_name = _ablation_run_name(experiment=experiment, selected_model=selected_model)
        run_name = run_base_name if overwrite else _unique_run_name(output_dir, run_base_name)
        run_dir = output_dir / run_name
        row = _base_ablation_row(
            experiment=experiment,
            dataset_yaml=dataset_yaml,
            selected_model=selected_model,
            run_name=run_name,
            run_dir=run_dir,
            use_online_augmentation=use_online_aug,
        )

        try:
            if not dataset_yaml.exists():
                raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

            resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(dataset_yaml, output_dir)
            train_args = _build_ablation_train_args(
                base_train_args=base_train_args,
                online_aug_args=online_aug_args,
                offline_aug_args=offline_aug_args,
                use_online_augmentation=use_online_aug,
            )
            row["resolved_data_yaml"] = str(resolved_data_yaml)

            if dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                row.update(
                    {
                        "status": "dry_run",
                        "notes": "training skipped because dry_run=True",
                    }
                )
            else:
                train_result = _train_one_candidate(
                    weights=selected_model,
                    data_yaml=resolved_data_yaml,
                    output_dir=output_dir,
                    run_name=run_name,
                    train_args=train_args,
                )
                actual_run_dir = _resolve_run_dir(train_result, fallback=run_dir)
                row.update(
                    {
                        "status": "trained",
                        "run_dir": str(actual_run_dir),
                        "error_message": "",
                        **extract_training_metrics(actual_run_dir),
                    }
                )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error_message": str(exc),
                    "notes": "ablation experiment failed; remaining experiments can continue",
                }
            )
            rows.append(row)
            if not continue_on_error:
                raise
            continue

        rows.append(row)

    return pd.DataFrame(rows)


def train_final_model(
    selected_model: str,
    data_yaml: Path,
    output_dir: Path,
    run_name: str,
    train_args: dict,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train the final YOLO model after architecture and experiment are locked.

    Notebook 07 intentionally uses ``exp_D_full_pipeline`` for practical
    robustness, even if the validation ablation ranking put another experiment
    first. This helper trains one model only and records all important choices
    so the final selection override is visible in reports.

    Args:
        selected_model: Fixed YOLO checkpoint selected from prior notebooks.
        data_yaml: Final dataset YAML, expected to be ``data_exp_D_full_pipeline.yaml``.
        output_dir: Parent directory for final model runs.
        run_name: Human-readable final run name.
        train_args: Ultralytics training arguments plus notebook-only metadata.
        overwrite: Replace an existing run only when explicitly enabled.

    Returns:
        A summary dictionary with training status and weight paths.
    """
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not selected_model:
        raise ValueError("selected_model is required for final training.")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    selected_experiment = str(train_args.get("selected_experiment", "exp_D_full_pipeline"))
    base_run_name = run_name or _final_run_name(selected_model, selected_experiment)
    safe_run_name = base_run_name if overwrite else _unique_run_name(output_dir, base_run_name)
    run_dir = output_dir / safe_run_name
    resolved_data_yaml = _resolve_dataset_yaml_for_ultralytics(data_yaml, output_dir)

    result: dict[str, Any] = {
        "selected_model": selected_model,
        "selected_experiment": selected_experiment,
        "dataset_yaml": str(data_yaml),
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
            weights=selected_model,
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


def _train_one_candidate(
    weights: str,
    data_yaml: Path,
    output_dir: Path,
    run_name: str,
    train_args: dict,
) -> Any:
    """Run one Ultralytics training call with notebook-only keys removed."""
    from ultralytics import YOLO

    custom_keys = {
        "dry_run",
        "overwrite",
        "selected_experiment",
        "manual_final_selection_override",
        "override_reason",
    }
    yolo_args = {key: value for key, value in train_args.items() if key not in custom_keys and value is not None}
    yolo_args.update(
        {
            "data": str(data_yaml),
            "project": str(output_dir),
            "name": run_name,
            "exist_ok": bool(train_args.get("overwrite", False)),
        }
    )

    model = YOLO(weights)
    return model.train(**yolo_args)


def _resolve_dataset_yaml_for_ultralytics(data_yaml: Path, output_dir: Path) -> Path:
    """Create a runtime YAML with an absolute dataset root for Ultralytics.

    The repo YAMLs intentionally use paths relative to ``sign-detection``.
    Some Ultralytics versions resolve relative ``path`` values against their
    global datasets directory, so this ignored runtime copy prevents accidental
    training against the wrong folder. The helper always refreshes only this
    generated YAML because it is safe to overwrite.
    """
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
    """Atomically refresh a generated runtime YAML file."""
    path = Path(path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)
    temporary_path.replace(path)


def _candidate_name(candidate: dict | str) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("name") or Path(str(candidate.get("weights", ""))).stem)
    return Path(str(candidate)).stem


def _candidate_weights(candidate: dict | str) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("weights") or candidate.get("name"))
    return str(candidate)


def _candidate_run_name(candidate: dict | str) -> str:
    weights = Path(_candidate_weights(candidate)).name
    if weights in MODEL_RUN_NAMES:
        return MODEL_RUN_NAMES[weights]

    name = _candidate_name(candidate).replace(" ", "_")
    return f"{name}_expD"


def _ablation_run_name(experiment: str, selected_model: str) -> str:
    prefix = ABLATION_RUN_PREFIXES.get(experiment, experiment or "ablation")
    model_stem = Path(selected_model).stem.replace(".", "_")
    return f"{prefix}_{model_stem}"


def _final_run_name(selected_model: str, selected_experiment: str) -> str:
    model_stem = Path(selected_model).stem.replace(".", "_")
    experiment_label = selected_experiment.replace("exp_", "exp_").replace(" ", "_")
    return f"final_{model_stem}_{experiment_label}"


def _build_ablation_train_args(
    base_train_args: dict,
    online_aug_args: dict,
    offline_aug_args: dict,
    use_online_augmentation: bool,
) -> dict:
    """Attach online augmentation only for experiments B and D."""
    train_args = dict(base_train_args)
    if use_online_augmentation:
        train_args.update(online_aug_args)
    else:
        train_args.update(offline_aug_args)
    return train_args


def _unique_run_name(output_dir: Path, base_name: str) -> str:
    """Return a deterministic suffix when an existing run should be preserved."""
    output_dir = Path(output_dir)
    candidate = base_name
    suffix = 2
    while (output_dir / candidate).exists():
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


def _base_candidate_row(
    model_name: str,
    weights: str,
    run_name: str,
    run_dir: Path,
    data_yaml: Path,
    resolved_data_yaml: Path,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "weights": weights,
        "run_name": run_name,
        "status": "not_started",
        "data_yaml": str(data_yaml),
        "resolved_data_yaml": str(resolved_data_yaml),
        "run_dir": str(run_dir),
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fitness": None,
        "training_time": None,
        "best_weights_path": "",
        "last_weights_path": "",
        "model_size_mb": None,
        "fps": None,
        "avg_latency_ms": None,
        "num_benchmark_images": None,
        "error_message": "",
        "notes": "",
    }


def _base_ablation_row(
    experiment: str,
    dataset_yaml: Path,
    selected_model: str,
    run_name: str,
    run_dir: Path,
    use_online_augmentation: bool,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "dataset_yaml": str(dataset_yaml),
        "selected_model": selected_model,
        "use_online_augmentation": use_online_augmentation,
        "run_name": run_name,
        "status": "not_started",
        "resolved_data_yaml": "",
        "run_dir": str(run_dir),
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "fitness": None,
        "training_time": None,
        "best_weights_path": "",
        "last_weights_path": "",
        "model_size_mb": None,
        "fps": None,
        "avg_latency_ms": None,
        "num_benchmark_images": None,
        "error_message": "",
        "notes": "",
    }
