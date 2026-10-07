"""Score one model answer against the manual ground truth."""

from __future__ import annotations

from .boxes import CONVENTIONS, EVAL_SIZE, Box, Match, iou_matrix, raw_to_box, score
from .parsing import extract_boxes

PRIMARY_THRESHOLD = 0.5
# A sweep result must beat the declared convention by this much F1 before it is flagged.
MISMATCH_MARGIN = 0.10


def _key(threshold: float) -> str:
    return f"{threshold:g}"


def evaluate_boxes(
    gt: list[Box],
    pred: list[Box],
    thresholds: tuple[float, ...],
    border_margin: float,
) -> tuple[dict, Match]:
    """Strict scores at every threshold plus border-tolerant scores at the primary threshold."""
    if PRIMARY_THRESHOLD not in thresholds:
        raise ValueError(f"thresholds must include {PRIMARY_THRESHOLD}")
    ious = iou_matrix(gt, pred)
    row: dict = {"n_gt": len(gt), "n_pred": len(pred)}
    primary_match = None
    for t in thresholds:
        s, match = score(gt, pred, t, None, ious=ious)
        row[f"p@{_key(t)}"] = s.precision
        row[f"r@{_key(t)}"] = s.recall
        row[f"f1@{_key(t)}"] = s.f1
        if t == PRIMARY_THRESHOLD:
            primary_match = match
            row.update(tp=s.tp, fp=s.fp, fn=s.fn, mean_matched_iou=s.mean_matched_iou, mean_best_iou=s.mean_best_iou)
    tolerant, _ = score(gt, pred, PRIMARY_THRESHOLD, border_margin, ious=ious)
    row["p_border@0.5"] = tolerant.precision
    row["f1_border@0.5"] = tolerant.f1
    row["border_ignored"] = tolerant.ignored
    assert primary_match is not None
    return row, primary_match


def evaluate_text(
    text: str,
    gt: list[Box],
    sent_size: tuple[int, int],
    convention: str,
    thresholds: tuple[float, ...],
    border_margin: float,
    eval_size: tuple[int, int] = EVAL_SIZE,
) -> tuple[dict, list[Box], Match]:
    parsed = extract_boxes(text)
    converted = [raw_to_box(r, convention, sent_size, eval_size) for r in parsed.boxes]
    pred = [b for b in converted if b is not None]
    row, match = evaluate_boxes(gt, pred, thresholds, border_margin)
    row.update(parse_mode=parsed.mode, n_raw=len(parsed.boxes), n_dropped=len(converted) - len(pred))

    # Diagnostic: would a different coordinate convention have scored much better?
    sweep: dict[str, float] = {}
    for other in CONVENTIONS:
        boxes = [b for b in (raw_to_box(r, other, sent_size, eval_size) for r in parsed.boxes) if b is not None]
        s, _ = score(gt, boxes, PRIMARY_THRESHOLD, None)
        sweep[other] = s.f1
    alternatives = {c: f for c, f in sweep.items() if c != convention}
    best_alt = max(alternatives, key=alternatives.get)
    row["best_alt_convention"] = best_alt
    row["best_alt_f1"] = alternatives[best_alt]
    row["convention_mismatch"] = alternatives[best_alt] > row["f1@0.5"] + MISMATCH_MARGIN
    # Diagnostic upper view: F1 if the model's own coordinate convention had been accepted. Picks the best of
    # six conventions per call, which only inflates the score when none is right; with dozens of boxes a wrong
    # convention scores near zero by chance, so this separates "wrong format" from "wrong place".
    row["f1_any_convention@0.5"] = max(row["f1@0.5"], alternatives[best_alt])
    return row, pred, match
