"""Small plotting utilities for factory sign detection reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def plot_class_distribution(class_distribution: pd.DataFrame, output_path: Path) -> Path:
    """Save a compact object-count bar chart for Notebook 01."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(class_distribution["class_name"], class_distribution["object_count"], color="#2f6f73")
    ax.set_title("Input Object Count by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Object count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_split_class_distribution(class_distribution: pd.DataFrame, output_path: Path) -> Path:
    """Save a grouped bar chart of object counts by split and class."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = class_distribution.pivot(index="class_name", columns="split", values="object_count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#2f6f73", "#d08c3c", "#7c4d79"])
    ax.set_title("Object Count by Split and Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Object count")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)
    return output_path


def plot_split_no_sign_ratio(no_sign_distribution: pd.DataFrame, output_path: Path) -> Path:
    """Save a no-sign ratio bar chart by split."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(no_sign_distribution["split"], no_sign_distribution["no_sign_ratio"], color="#4d7896")
    ax.set_ylim(0, 1)
    ax.set_title("No-Sign Image Ratio by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("No-sign ratio")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_bbox_area_by_class(bbox_with_split: pd.DataFrame, output_path: Path) -> Path:
    """Save a simple boxplot of normalized bbox area by class."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    if bbox_with_split.empty:
        ax.text(0.5, 0.5, "No bounding boxes available", ha="center", va="center")
        ax.axis("off")
    else:
        labels = []
        data = []
        for class_name, group in bbox_with_split.groupby("class_name", sort=True):
            labels.append(class_name)
            data.append(group["box_area_norm"].astype(float).values)
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_yscale("log")
        ax.set_title("BBox Area by Class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Normalized bbox area")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_bbox_center_heatmap(bbox_with_split: pd.DataFrame, output_path: Path) -> Path:
    """Save a bbox-center heatmap in normalized image coordinates."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    if bbox_with_split.empty:
        ax.text(0.5, 0.5, "No bounding boxes available", ha="center", va="center")
        ax.axis("off")
    else:
        heat = ax.hist2d(
            bbox_with_split["x_center_norm"].astype(float),
            bbox_with_split["y_center_norm"].astype(float),
            bins=20,
            range=[[0, 1], [0, 1]],
            cmap="viridis",
        )
        fig.colorbar(heat[3], ax=ax, label="Box count")
        ax.invert_yaxis()
        ax.set_title("BBox Center Location Heatmap")
        ax.set_xlabel("x center")
        ax.set_ylabel("y center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_image_quality_by_split(image_quality_summary: pd.DataFrame, output_path: Path) -> Path:
    """Save brightness and blur summaries by split."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if image_quality_summary.empty:
        axes[0].text(0.5, 0.5, "No image quality data", ha="center", va="center")
        axes[1].axis("off")
    else:
        axes[0].bar(image_quality_summary["split"], image_quality_summary["mean_brightness_mean"], color="#6d8f52")
        axes[0].set_title("Mean Brightness")
        axes[0].set_ylabel("0-255 grayscale")
        axes[1].bar(image_quality_summary["split"], image_quality_summary["mean_blur_score_laplacian"], color="#8e6c9f")
        axes[1].set_title("Mean Laplacian Blur Score")
        axes[1].set_ylabel("higher = sharper")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _draw_boxes(image: Image.Image, boxes: pd.DataFrame) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = canvas.size
    colors = ["#2f6f73", "#d08c3c", "#7c4d79", "#4d7896"]
    for _, row in boxes.iterrows():
        box_w = float(row["width_norm"]) * width
        box_h = float(row["height_norm"]) * height
        x = float(row["x_center_norm"]) * width
        y = float(row["y_center_norm"]) * height
        left = x - box_w / 2
        top = y - box_h / 2
        right = x + box_w / 2
        bottom = y + box_h / 2
        color = colors[int(row["class_id"]) % len(colors)]
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left + 2, max(0, top - 12)), str(row["class_name"]), fill=color, font=font)
    return canvas


def plot_sample_annotations_grid(
    split_df: pd.DataFrame,
    bbox_with_split: pd.DataFrame,
    output_path: Path,
    max_samples: int = 12,
) -> Path:
    """Save one compact sample grid with labeled, no-sign, and tiny-sign examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled = split_df[split_df["num_objects"].fillna(0).astype(int) > 0]
    no_sign = split_df[split_df["is_no_sign"].fillna(False).astype(bool)]
    tiny_names: list[str] = []
    if not bbox_with_split.empty:
        tiny = bbox_with_split[
            (bbox_with_split["box_area_norm"] < 0.0005)
            | (bbox_with_split["box_width_px"] < 8)
            | (bbox_with_split["box_height_px"] < 8)
        ]
        tiny_names = tiny["image_name"].drop_duplicates().head(4).tolist()

    sample_rows = pd.concat(
        [
            labeled.groupby("split", group_keys=False).head(2),
            no_sign.head(3),
            split_df[split_df["image_name"].isin(tiny_names)],
        ],
        ignore_index=True,
    ).drop_duplicates("image_name").head(max_samples)

    if sample_rows.empty:
        Image.new("RGB", (900, 240), "white").save(output_path)
        return output_path

    thumbs = []
    for _, row in sample_rows.iterrows():
        image_path = Path(str(row["target_image_path"]))
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            boxes = bbox_with_split[bbox_with_split["image_name"] == row["image_name"]]
            drawn = _draw_boxes(image, boxes)
            drawn.thumbnail((260, 180))
            tile = Image.new("RGB", (280, 220), "white")
            tile.paste(drawn, ((280 - drawn.width) // 2, 10))
            draw = ImageDraw.Draw(tile)
            draw.text((10, 194), f"{row['split']} | {row['image_name']}", fill="#222222")
            thumbs.append(tile)

    if not thumbs:
        Image.new("RGB", (900, 240), "white").save(output_path)
        return output_path

    cols = min(3, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 280, rows * 220), "white")
    for index, tile in enumerate(thumbs):
        x = (index % cols) * 280
        y = (index // cols) * 220
        grid.paste(tile, (x, y))
    grid.save(output_path)
    return output_path
