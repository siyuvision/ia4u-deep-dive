"""Speed and price figures for model calls, and which models are not beaten on both quality and cost."""

from __future__ import annotations

import statistics

from .openrouter import answer_latency_s


def _ms_to_s(value) -> float | None:
    return float(value) / 1000.0 if isinstance(value, (int, float)) and value >= 0 else None


def speed_fields(rec: dict) -> dict:
    """Per-call speed columns from one stored raw record.

    latency_s      seconds of the request that produced the answer (no failed attempts, no backoff sleeps)
    wall_s         seconds from the first attempt to the answer, including retries
    tok_per_s      completion tokens (reasoning included) / latency_s: end-to-end, not decode-only speed
    ttft_s         time to first token as reported by OpenRouter's /generation record (None if unavailable)
    gen_s          generation time as reported by OpenRouter's /generation record (None if unavailable)
    """
    call = rec["call"]
    latency = answer_latency_s(call)
    completion = (rec.get("usage") or {}).get("completion_tokens")
    generation = rec.get("generation") or {}
    return {
        "latency_s": round(latency, 2) if latency is not None else None,
        "wall_s": round(call["latency_s"], 2) if isinstance(call.get("latency_s"), (int, float)) else None,
        "attempts": len(call.get("attempts") or []) or None,
        "tok_per_s": round(completion / latency, 1) if latency and isinstance(completion, (int, float)) else None,
        "ttft_s": _ms_to_s(generation.get("latency")),
        "gen_s": _ms_to_s(generation.get("generation_time")),
    }


def numbers(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))]


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def cost_per_true_positive(costs: list[float], true_positives: list[float]) -> float | None:
    """Mean billed cost of one call divided by mean matched boxes per call (IoU 0.5); None if nothing matched."""
    if not costs or not true_positives:
        return None
    tp = statistics.mean(true_positives)
    return statistics.mean(costs) / tp if tp > 0 else None


def list_price_per_million(entry: dict | None) -> tuple[float, float] | None:
    """(input, output) USD per million tokens from a models snapshot entry."""
    if not entry:
        return None
    try:
        p = entry["pricing_usd_per_token"]
        return float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6
    except (KeyError, TypeError, ValueError):
        return None


def non_dominated(points: dict[str, tuple[float, float]]) -> list[str]:
    """Names whose (quality, cost) point no other point beats: nobody has quality >= and cost <= with one strict.

    Higher quality is better, lower cost (or time) is better. Ties on both axes do not dominate each other.
    """
    keep = []
    for name, (q, c) in points.items():
        beaten = any(
            (q2 >= q and c2 <= c) and (q2 > q or c2 < c)
            for other, (q2, c2) in points.items()
            if other != name
        )
        if not beaten:
            keep.append(name)
    return keep
