"""Overlay of ground truth and one set of predictions on the input image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .boxes import EVAL_SIZE, Box, Match

GT_COLOR = (0, 170, 0)
MATCHED_COLOR = (0, 110, 255)
UNMATCHED_COLOR = (255, 60, 0)


def draw_overlay(
    image_path: Path,
    gt: list[Box],
    pred: list[Box],
    match: Match,
    out_path: Path,
    title: str,
    eval_size: tuple[int, int] = EVAL_SIZE,
) -> None:
    """Green = ground truth, blue = prediction matched at IoU 0.5, red = unmatched prediction."""
    image = Image.open(image_path).convert("RGB").resize(eval_size)
    draw = ImageDraw.Draw(image)
    for box in gt:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=GT_COLOR, width=3)
    for i, box in enumerate(pred):
        color = MATCHED_COLOR if match.pred_to_gt[i] >= 0 else UNMATCHED_COLOR
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=2)
    draw.rectangle([0, 0, 640, 22], fill=(255, 255, 255))
    draw.text((6, 5), f"{title} | green=GT blue=matched@0.5 red=unmatched", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
