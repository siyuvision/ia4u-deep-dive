"""Score the stored raw answers of one run against the manual ground truth.

Writes, under results/<run-id>/: scores.csv (one row per call), summary.md (one row per model),
parsed/ (predicted boxes in the 1164 x 900 evaluation space, Roboflow center format) and
overlays/ (ground truth vs predictions on the image).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "src"))
from mllm_bbox.boxes import EVAL_SIZE, Box, boxes_from_center_json  # noqa: E402
from mllm_bbox.efficiency import (  # noqa: E402
    cost_per_true_positive,
    list_price_per_million,
    mean_or_none,
    median_or_none,
    non_dominated,
    numbers,
    speed_fields,
)
from mllm_bbox.evaluation import evaluate_text  # noqa: E402
from mllm_bbox.openrouter import embedded_error  # noqa: E402
from mllm_bbox.render import draw_overlay  # noqa: E402

DEFAULT_IMAGE = PROJECT / "data" / "input" / "kidney-he-case01-1024x791.jpg"
DEFAULT_GT = PROJECT / "data" / "annotations" / "gt-manual.json"
DEFAULT_PRICING = PROJECT / "sources" / "openrouter-models-20261007.json"

def to_center_json(boxes: list[Box]) -> dict:
    return {
        "image": {"width": EVAL_SIZE[0], "height": EVAL_SIZE[1]},
        "predictions": [
            {
                "x": round((b.x1 + b.x2) / 2, 1),
                "y": round((b.y1 + b.y2) / 2, 1),
                "width": round(b.w, 1),
                "height": round(b.h, 1),
                "class": "renal tubule",
            }
            for b in boxes
        ],
    }


def fmt(values: list[float], digits: int = 3) -> str:
    return f"{statistics.mean(values):.{digits}f}" if values else "-"


def fmt_range(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{statistics.mean(values):.3f} [{min(values):.3f}-{max(values):.3f}]"


def fmt_value(value: float | None, digits: int = 1, prefix: str = "") -> str:
    return "-" if value is None else f"{prefix}{value:.{digits}f}"


def fmt_median_range(values: list[float], digits: int = 1) -> str:
    if not values:
        return "-"
    return f"{statistics.median(values):.{digits}f} [{min(values):.{digits}f}-{max(values):.{digits}f}]"


def fmt_usd(value: float | None) -> str:
    """Dollar amounts span 1e-5 to 1e0 here, so show enough digits to tell models apart."""
    if value is None:
        return "-"
    if value < 0.001:
        return f"${value:.5f}"
    return f"${value:.4f}" if value < 0.1 else f"${value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--thresholds", default="0.3,0.5,0.75")
    parser.add_argument("--border-margin", type=float, default=10.0, help="px; unmatched predictions this close to the image border are ignored in the border-tolerant score")
    parser.add_argument(
        "--convention",
        default="xyxy_norm1000",
        help="coordinate convention the prompt asked for, for every model whose config entry has no `convention` of its own",
    )
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING, help="models snapshot written by check_models.py, for list prices")
    args = parser.parse_args(argv)

    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    run_dir = args.results_dir / args.run_id
    raw_files = sorted((run_dir / "raw").glob("*.json"))
    if not raw_files:
        print(f"no raw answers in {run_dir / 'raw'}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    gt = boxes_from_center_json(json.loads(args.gt.read_text(encoding="utf-8")))
    with Image.open(args.image) as im:
        sent_size = im.size

    rows: list[dict] = []
    for path in raw_files:
        rec = json.loads(path.read_text(encoding="utf-8"))
        key, repeat = rec["model"]["key"], rec["repeat"]
        convention = rec["model"].get("convention") or args.convention  # what this call's prompt asked for
        base = {
            "model": key,
            "convention": convention,
            "model_id": rec["model"]["id"],
            "label": rec["model"]["label"],
            "repeat": repeat,
            "call_ok": bool(rec["call"]["ok"]) and embedded_error(rec["call"].get("body")) is None,
            "http_status": rec["call"]["status"],
            "finish_reason": rec.get("finish_reason"),
            **speed_fields(rec),
            **{k: rec["usage"].get(k) for k in ("provider", "served_model", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd")},
        }
        if not base["call_ok"]:
            rows.append(base)
            continue
        metrics, pred, match = evaluate_text(rec["text"], gt, sent_size, convention, thresholds, args.border_margin)
        rows.append({**base, **metrics})
        parsed_path = run_dir / "parsed" / f"{key}__r{repeat}.json"
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(json.dumps(to_center_json(pred), indent=2) + "\n", encoding="utf-8")
        if not args.no_overlays:
            draw_overlay(args.image, gt, pred, match, run_dir / "overlays" / f"{key}__r{repeat}.png", f"{key} r{repeat}")

    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    pricing_by_id: dict[str, dict] = {}
    if args.pricing.is_file():
        pricing_by_id = {m["id"]: m for m in json.loads(args.pricing.read_text(encoding="utf-8"))["models"]}

    # Per-model summary. A failed call (transport) is excluded from means; an unusable answer counts as zero.
    summary = []
    for key in dict.fromkeys(r["model"] for r in rows):
        group = [r for r in rows if r["model"] == key]
        scored = [r for r in group if r.get("call_ok") and "f1@0.5" in r]
        flags = []
        if len(scored) < len(group):
            flags.append(f"{len(group) - len(scored)} call(s) failed")
        flags += sorted({f"parse:{r['parse_mode']}" for r in scored if r.get("parse_mode") != "json"})
        if any(r.get("finish_reason") == "length" for r in scored):
            flags.append("truncated")
        mismatch = sorted({r["best_alt_convention"] for r in scored if r.get("convention_mismatch")})
        if mismatch:
            flags.append("convention? " + "/".join(mismatch))
        # Speed and price cover every call that returned an answer, including unusable ones: they cost time and money.
        answered = [r for r in group if r.get("call_ok")]
        costs = numbers(answered, "cost_usd")
        latencies = numbers(answered, "latency_s")
        mean_cost = mean_or_none(costs)
        summary.append(
            {
                "model": key,
                "model_id": group[0].get("model_id"),
                "label": group[0]["label"],
                "mean_f1": statistics.mean([r["f1@0.5"] for r in scored]) if scored else None,
                "mean_cost": mean_cost,
                "median_latency": median_or_none(latencies),
                "latencies": latencies,
                "providers": sorted({r["provider"] for r in answered if r.get("provider")}),
                "completion_tokens": mean_or_none(numbers(answered, "completion_tokens")),
                "reasoning_tokens": mean_or_none(numbers(answered, "reasoning_tokens")),
                "tok_per_s": median_or_none(numbers(answered, "tok_per_s")),
                "ttft_s": median_or_none(numbers(answered, "ttft_s")),
                "ttft_n": len(numbers(answered, "ttft_s")),
                "n_answered": len(answered),
                "total_cost": sum(costs) if costs else None,
                "cost_per_tp": cost_per_true_positive(numbers(answered, "cost_usd"), numbers(scored, "tp")),
                "list_price": list_price_per_million(pricing_by_id.get(group[0].get("model_id"))),
                "runs": f"{len(scored)}/{len(group)}",
                "n_pred": fmt([r["n_pred"] for r in scored], 1),
                "f1": fmt_range([r["f1@0.5"] for r in scored]),
                "p": fmt([r["p@0.5"] for r in scored]),
                "r": fmt([r["r@0.5"] for r in scored]),
                "f1_75": fmt([r["f1@0.75"] for r in scored if "f1@0.75" in r]),
                "f1_30": fmt([r["f1@0.3"] for r in scored if "f1@0.3" in r]),
                "best_iou": fmt([r["mean_best_iou"] for r in scored]),
                "f1_border": fmt([r["f1_border@0.5"] for r in scored]),
                "f1_any": fmt_range([r["f1_any_convention@0.5"] for r in scored if "f1_any_convention@0.5" in r]),
                "flags": "; ".join(flags) or "-",
                "_sort": statistics.mean([r["f1@0.5"] for r in scored]) if scored else -1.0,
            }
        )
    summary.sort(key=lambda s: -s["_sort"])

    header = (
        f"# Run {args.run_id}: {len(gt)} ground-truth boxes, IoU threshold 0.5 unless stated\n\n"
        "F1/P/R are strict (every unmatched prediction is a false positive). 'F1 border-tolerant' ignores unmatched "
        f"predictions within {args.border_margin:g} px of the image border. 'Best IoU' is the mean over ground-truth boxes of the "
        "best IoU with any prediction. F1@0.5 shows mean [min-max] over repeats. 'F1 any convention' is a diagnostic, not the main "
        "score: per call, the best F1@0.5 over six coordinate conventions (x/y order, 0-1000 / pixels / 0-1), i.e. what the model "
        "would score if its own format were accepted. A large gap to F1@0.5 means the model located tubules but did not follow the "
        "requested coordinate format.\n\n"
    )
    lines = [
        "| Model | Runs | Pred | F1@0.5 | P@0.5 | R@0.5 | F1@0.75 | F1@0.3 | Best IoU | F1 border-tolerant | F1 any convention | Flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['label']} | {s['runs']} | {s['n_pred']} | {s['f1']} | {s['p']} | {s['r']} | {s['f1_75']} | {s['f1_30']} | "
            f"{s['best_iou']} | {s['f1_border']} | {s['f1_any']} | {s['flags']} |"
        )

    # Speed and price: measured on this run only (one image, a few calls per model), not a general benchmark.
    if summary:
        lines += [
            "",
            "## Speed and price",
            "",
            "Measured on this run's calls (all answered calls, including unusable answers). 'Latency' is the request that produced the "
            "answer, without retries, median [min-max] s; calls ran up to "
            "4 at a time, different models in parallel. 'Out tok/s' is completion tokens (reasoning included) divided by that latency, so it is "
            "end-to-end speed, not decode speed. 'TTFT' is OpenRouter's own time-to-first-token record (median); for reasoning models it can "
            "include thinking time before the first token. 'List $/M' is the model's listed "
            "input/output price; '$/call' is what OpenRouter billed. '$ per match' is mean $/call divided by mean boxes matched at IoU 0.5.\n",
            "| Model | List $/M in/out | Provider | Out tok (reasoning) | Latency s | TTFT s | Out tok/s | $/call | Total $ | $ per match | F1@0.5 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in summary:
            price = f"{s['list_price'][0]:g} / {s['list_price'][1]:g}" if s["list_price"] else "-"
            tokens = fmt_value(s["completion_tokens"], 0)
            if s["reasoning_tokens"] is not None:
                tokens += f" ({s['reasoning_tokens']:.0f})"
            ttft = fmt_value(s["ttft_s"], 1)
            if 0 < s["ttft_n"] < s["n_answered"]:  # OpenRouter had no record for some calls: say how many the median rests on
                ttft += f" (n={s['ttft_n']}/{s['n_answered']})"
            lines.append(
                f"| {s['label']} | {price} | {', '.join(s['providers']) or '-'} | {tokens} | {fmt_median_range(s['latencies'])} | "
                f"{ttft} | {fmt_value(s['tok_per_s'], 0)} | {fmt_usd(s['mean_cost'])} | {fmt_usd(s['total_cost'])} | "
                f"{fmt_usd(s['cost_per_tp'])} | {fmt_value(s['mean_f1'], 3)} |"
            )
        by_cost = {s["label"]: (s["mean_f1"], s["mean_cost"]) for s in summary if s["mean_f1"] is not None and s["mean_cost"] is not None}
        by_time = {s["label"]: (s["mean_f1"], s["median_latency"]) for s in summary if s["mean_f1"] is not None and s["median_latency"] is not None}
        lines.append("")
        if by_cost:
            lines.append(f"- Not beaten on F1 vs $/call (no other model is at least as accurate and cheaper): {', '.join(non_dominated(by_cost))}.")
        if by_time:
            lines.append(f"- Not beaten on F1 vs median latency (no other model is at least as accurate and faster): {', '.join(non_dominated(by_time))}.")

    text = header + "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {run_dir / 'summary.md'}, scores.csv, parsed/, overlays/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
