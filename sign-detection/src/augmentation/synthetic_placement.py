"""Synthetic sign placement utilities that must update labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .offline_common import empty_report, prepare_output_dirs, report_row


def generate_synthetic_placement_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    ratio: float,
    config: dict | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Return a safe placeholder report for synthetic sign placement.

    Synthetic copy-paste changes object geometry and therefore must update
    labels. It is intentionally disabled until a reliable placement workflow is
    implemented.
    """
    del images_dir, labels_dir, config, seed, overwrite
    prepare_output_dirs(output_images_dir, output_labels_dir)
    if ratio <= 0:
        return pd.DataFrame(
            [
                report_row(
                    "synthetic_placement",
                    "",
                    "",
                    "",
                    "",
                    "skipped",
                    "synthetic placement disabled because ratio is 0.0",
                    0,
                    0,
                )
            ]
        )
    report = empty_report()
    return pd.concat(
        [
            report,
            pd.DataFrame(
                [
                    report_row(
                        "synthetic_placement",
                        "",
                        "",
                        "",
                        "",
                        "skipped",
                        "synthetic placement placeholder only; label-safe copy-paste not implemented yet",
                        0,
                        0,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
