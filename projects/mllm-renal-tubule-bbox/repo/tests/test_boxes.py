import json
from pathlib import Path

import pytest

from mllm_bbox.boxes import (
    Box,
    boxes_from_center_json,
    iou,
    iou_matrix,
    match_greedy,
    raw_to_box,
    score,
)

SENT = (1024, 791)
PROJECT = Path(__file__).resolve().parents[2]


def approx(v):
    return pytest.approx(v, abs=1e-9)


def test_iou_hand_computed_values():
    assert iou(Box(0, 0, 100, 100), Box(0, 0, 100, 100)) == approx(1.0)
    assert iou(Box(0, 0, 100, 100), Box(500, 500, 600, 600)) == 0.0
    assert iou(Box(0, 0, 100, 100), Box(100, 0, 200, 100)) == 0.0  # touching edges
    assert iou(Box(0, 0, 100, 100), Box(50, 0, 150, 100)) == approx(1 / 3)  # half overlap
    assert iou(Box(0, 0, 100, 100), Box(25, 25, 75, 75)) == approx(0.25)  # containment
    assert iou(Box(0, 0, 10, 10), Box(5, 5, 15, 20)) == approx(25 / 225)  # inter 25, union 100 + 150 - 25


def test_center_json_to_corners():
    doc = {"predictions": [{"x": 106, "y": 102.5, "width": 116, "height": 99}]}
    (box,) = boxes_from_center_json(doc)
    assert (box.x1, box.y1, box.x2, box.y2) == (48, 53, 164, 152)


def test_raw_to_box_conventions():
    full = raw_to_box([0, 0, 1000, 1000], "xyxy_norm1000", SENT)
    assert (full.x1, full.y1, full.x2, full.y2) == (0, 0, 1164, 900)
    # yxyx: values are [y1, x1, y2, x2]
    b = raw_to_box([100, 200, 300, 400], "yxyx_norm1000", SENT)
    assert (b.x1, b.y1, b.x2, b.y2) == (approx(232.8), approx(90.0), approx(465.6), approx(270.0))
    # pixels of the image that was sent are rescaled into the evaluation space
    px = raw_to_box([0, 0, 1024, 791], "xyxy_px", SENT)
    assert (px.x2, px.y2) == (approx(1164), approx(900))
    unit = raw_to_box([0.5, 0.5, 1.0, 1.0], "xyxy_norm1", SENT)
    assert (unit.x1, unit.y1, unit.x2, unit.y2) == (approx(582), approx(450), approx(1164), approx(900))


def test_raw_to_box_sorts_clips_and_drops_degenerate():
    reversed_box = raw_to_box([500, 500, 100, 100], "xyxy_norm1000", SENT)
    assert reversed_box.x1 < reversed_box.x2 and reversed_box.y1 < reversed_box.y2
    clipped = raw_to_box([-50, -50, 2000, 2000], "xyxy_norm1000", SENT)
    assert (clipped.x1, clipped.y1, clipped.x2, clipped.y2) == (0, 0, 1164, 900)
    assert raw_to_box([10, 10, 10, 50], "xyxy_norm1000", SENT) is None
    assert raw_to_box([1100, 1100, 1200, 1200], "xyxy_norm1000", SENT) is None  # entirely outside after clipping
    with pytest.raises(ValueError):
        raw_to_box([0, 0, 1, 1], "cxcywh", SENT)


def test_greedy_matching_is_one_to_one_highest_iou_first():
    model = [Box(0, 0, 100, 100), Box(250, 250, 350, 350)]
    mine = [Box(0, 0, 100, 100), Box(20, 0, 120, 100)]  # IoU 1.0 and 80/120 with model[0]
    ious = iou_matrix(mine, model)
    match = match_greedy(ious, len(model), 0.5)
    assert match.gt_to_pred == [0, -1]
    assert match.pred_to_gt == [0, -1]
    assert match_greedy(ious, len(model), 0.95).gt_to_pred == [0, -1]
    assert match_greedy([[0.5]], 1, 0.5).gt_to_pred == [0]  # threshold is inclusive
    assert match_greedy([[0.0]], 1, 0.05).gt_to_pred == [-1]  # zero overlap never matches


def test_score_counts_and_rates():
    gt = [Box(0, 0, 100, 100), Box(200, 200, 300, 300), Box(400, 400, 500, 500)]
    pred = [Box(0, 0, 100, 100), Box(205, 200, 305, 300), Box(700, 700, 800, 800)]
    s, _ = score(gt, pred, 0.5)
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    assert s.precision == approx(2 / 3) and s.recall == approx(2 / 3)
    assert s.f1 == approx(4 / 6)  # 2tp / (2tp + fp + fn)
    assert s.mean_matched_iou == approx((1.0 + 95 / 105) / 2)
    assert s.mean_best_iou == approx((1.0 + 95 / 105 + 0.0) / 3)


def test_score_without_predictions_is_all_zero_not_an_error():
    s, _ = score([Box(0, 0, 10, 10)], [], 0.5)
    assert (s.tp, s.fp, s.fn) == (0, 0, 1)
    assert s.precision == 0.0 and s.recall == 0.0 and s.f1 == 0.0 and s.mean_best_iou == 0.0


def test_border_tolerance_ignores_only_unmatched_border_predictions():
    gt = [Box(100, 100, 200, 200), Box(1000, 800, 1160, 898)]  # second GT sits near the border
    pred = [
        Box(100, 100, 200, 200),  # matches GT 0
        Box(1000, 800, 1160, 898),  # matches GT 1 and touches the border: stays a true positive
        Box(0, 300, 80, 400),  # unmatched, touches the left border: ignored when tolerant
        Box(400, 400, 500, 500),  # unmatched, interior: always a false positive
    ]
    strict, _ = score(gt, pred, 0.5)
    tolerant, _ = score(gt, pred, 0.5, border_margin=10)
    assert (strict.tp, strict.fp, strict.ignored) == (2, 2, 0)
    assert (tolerant.tp, tolerant.fp, tolerant.ignored) == (2, 1, 1)
    assert tolerant.precision > strict.precision and tolerant.recall == strict.recall


def test_manual_ground_truth_is_self_consistent():
    path = PROJECT / "data" / "annotations" / "gt-manual.json"
    if not path.is_file():
        pytest.skip("ground truth is not tracked by git; place it in data/annotations/")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["image"] == {"width": 1164, "height": 900}
    gt = boxes_from_center_json(doc)
    assert len(gt) == 37
    assert all(0 <= b.x1 < b.x2 <= 1164 and 0 <= b.y1 < b.y2 <= 900 for b in gt)
    s, _ = score(gt, gt, 0.95)
    assert s.f1 == approx(1.0) and s.mean_matched_iou == approx(1.0)
