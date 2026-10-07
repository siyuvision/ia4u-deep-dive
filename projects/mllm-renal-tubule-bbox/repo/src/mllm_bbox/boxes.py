"""Box geometry, coordinate conventions, matching and scoring.

Everything is scored in the evaluation space: the 1164 x 900 pixel space that the manual ground
truth is drawn in. Model output is converted into that space first.
"""

from __future__ import annotations

from dataclasses import dataclass

EVAL_SIZE = (1164, 900)

# Convention = axis order + unit. The prompt asks for xyxy_norm1000; the others are only used by the
# diagnostic sweep that detects a model answering in its own native format.
CONVENTIONS = (
    "xyxy_norm1000",
    "yxyx_norm1000",
    "xyxy_px",
    "yxyx_px",
    "xyxy_norm1",
    "yxyx_norm1",
)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @staticmethod
    def from_center(cx: float, cy: float, w: float, h: float) -> "Box":
        return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def boxes_from_center_json(doc: dict) -> list[Box]:
    """Roboflow-style document: predictions with center x, y and width, height."""
    return [Box.from_center(p["x"], p["y"], p["width"], p["height"]) for p in doc["predictions"]]


def raw_to_box(
    raw: list[float],
    convention: str,
    sent_size: tuple[int, int],
    eval_size: tuple[int, int] = EVAL_SIZE,
) -> Box | None:
    """Convert one raw 4-number box into the evaluation space; None if it is degenerate after clipping."""
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}")
    order, unit = convention.split("_")
    a, b, c, d = raw
    if order == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
    else:
        y1, x1, y2, x2 = a, b, c, d
    ew, eh = eval_size
    if unit == "norm1000":
        sx, sy = ew / 1000, eh / 1000
    elif unit == "norm1":
        sx, sy = ew, eh
    else:  # px of the image that was sent to the model
        sx, sy = ew / sent_size[0], eh / sent_size[1]
    lo_x, hi_x = sorted((x1 * sx, x2 * sx))
    lo_y, hi_y = sorted((y1 * sy, y2 * sy))
    box = Box(min(max(lo_x, 0.0), ew), min(max(lo_y, 0.0), eh), min(max(hi_x, 0.0), ew), min(max(hi_y, 0.0), eh))
    if box.w <= 0 or box.h <= 0:
        return None
    return box


def iou(a: Box, b: Box) -> float:
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(gt: list[Box], pred: list[Box]) -> list[list[float]]:
    """Rows are ground-truth boxes, columns are predictions."""
    return [[iou(g, p) for p in pred] for g in gt]


def touches_border(box: Box, margin: float, eval_size: tuple[int, int] = EVAL_SIZE) -> bool:
    return box.x1 <= margin or box.y1 <= margin or box.x2 >= eval_size[0] - margin or box.y2 >= eval_size[1] - margin


@dataclass
class Match:
    gt_to_pred: list[int]
    pred_to_gt: list[int]


def match_greedy(ious: list[list[float]], n_pred: int, threshold: float) -> Match:
    """One-to-one matching, highest IoU first. IoU must be > 0 and >= threshold."""
    gt_to_pred = [-1] * len(ious)
    pred_to_gt = [-1] * n_pred
    pairs = [(v, g, p) for g, row in enumerate(ious) for p, v in enumerate(row) if v > 0 and v >= threshold]
    pairs.sort(key=lambda t: -t[0])
    for _, g, p in pairs:
        if gt_to_pred[g] < 0 and pred_to_gt[p] < 0:
            gt_to_pred[g] = p
            pred_to_gt[p] = g
    return Match(gt_to_pred, pred_to_gt)


@dataclass
class Score:
    threshold: float
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    ignored: int  # unmatched predictions touching the image border, excluded from fp when border_margin is set
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    mean_best_iou: float  # threshold-independent: mean over ground truth of the best IoU with any prediction


def score(
    gt: list[Box],
    pred: list[Box],
    threshold: float,
    border_margin: float | None = None,
    eval_size: tuple[int, int] = EVAL_SIZE,
    ious: list[list[float]] | None = None,
) -> tuple[Score, Match]:
    """Score predictions against ground truth.

    border_margin=None counts every unmatched prediction as a false positive (strict).
    With a margin, an unmatched prediction whose box lies within that many pixels of the image
    border is ignored, because the ground truth deliberately omits tubules cut by the border.
    """
    ious = ious if ious is not None else iou_matrix(gt, pred)
    match = match_greedy(ious, len(pred), threshold)
    tp = sum(1 for p in match.gt_to_pred if p >= 0)
    unmatched = [i for i, g in enumerate(match.pred_to_gt) if g < 0]
    ignored = 0
    if border_margin is not None:
        ignored = sum(1 for i in unmatched if touches_border(pred[i], border_margin, eval_size))
    fp = len(unmatched) - ignored
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    matched_ious = [ious[g][p] for g, p in enumerate(match.gt_to_pred) if p >= 0]
    best = [max(row) if row else 0.0 for row in ious]
    return (
        Score(
            threshold=threshold,
            n_gt=len(gt),
            n_pred=len(pred),
            tp=tp,
            fp=fp,
            fn=fn,
            ignored=ignored,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_matched_iou=sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
            mean_best_iou=sum(best) / len(best) if best else 0.0,
        ),
        match,
    )
