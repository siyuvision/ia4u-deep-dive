"""Extract raw 4-number boxes from free-form model text.

Models are asked for a bare JSON array but often add prose, a code fence or inline reasoning. The
parser prefers real JSON and only falls back to scanning for bracketed 4-number groups, and it says
which path it took so the report can show how clean each model's output was.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

BOX_KEYS = ("bbox_2d", "bbox", "box_2d", "box", "bounding_box")
CORNER_KEYS = (("x_min", "y_min", "x_max", "y_max"), ("xmin", "ymin", "xmax", "ymax"), ("x1", "y1", "x2", "y2"))
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_NUM = r"-?\d+(?:\.\d+)?"
_QUAD_RE = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]")


@dataclass
class ParseResult:
    boxes: list[list[float]]
    mode: str  # "json", "regex" or "none"


def _is_quad(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
    )


def _collect(obj, out: list[list[float]]) -> None:
    if _is_quad(obj):
        out.append([float(v) for v in obj])
    elif isinstance(obj, dict):
        for key in BOX_KEYS:
            if _is_quad(obj.get(key)):
                out.append([float(v) for v in obj[key]])
                return
        for keys in CORNER_KEYS:
            vals = [obj.get(k) for k in keys]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                out.append([float(v) for v in vals])
                return
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect(item, out)


def _json_candidates(text: str) -> tuple[list[list[list[float]]], list[list[float]]]:
    """Decode every JSON value in the text.

    Returns (structured, scattered): structured are decoded containers holding boxes (a list of boxes,
    an object with a boxes field, ...); scattered are bare [a, b, c, d] groups that each stand alone in
    the prose, which is how some models emit native box tokens.
    """
    decoder = json.JSONDecoder()
    structured: list[list[list[float]]] = []
    scattered: list[list[float]] = []
    i = 0
    while i < len(text):
        if text[i] in "[{":
            try:
                obj, end = decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if _is_quad(obj):
                scattered.append([float(v) for v in obj])
            else:
                boxes: list[list[float]] = []
                _collect(obj, boxes)
                if boxes:
                    structured.append(boxes)
            i = end
        else:
            i += 1
    return structured, scattered


def _repair_truncated(text: str) -> list[list[float]]:
    """Recover the complete leading items of an answer cut off by the token limit.

    Takes the text from the first '[' up to the last '}' or ']' and closes the array.
    Returns [] when that does not yield valid JSON (including when the text was complete anyway).
    """
    start = text.find("[")
    end = max(text.rfind("}"), text.rfind("]"))
    if start < 0 or end <= start:
        return []
    try:
        obj = json.loads(text[start : end + 1] + "]")
    except ValueError:
        return []
    boxes: list[list[float]] = []
    _collect(obj, boxes)
    return boxes


def extract_boxes(text: str | None) -> ParseResult:
    """Return the most complete JSON-encoded box list in the text, else regex-found quads, else none."""
    if not text:
        return ParseResult([], "none")
    cleaned = _THINK_RE.sub("", text)
    structured, scattered = _json_candidates(cleaned)
    repaired = _repair_truncated(cleaned)
    if structured or repaired:
        # Largest list wins; among equals the last one wins (a final answer follows any draft).
        pool = [(boxes, "json") for boxes in structured] + ([(repaired, "json_repaired")] if repaired else [])
        best = max(range(len(pool)), key=lambda k: (len(pool[k][0]), k))
        return ParseResult(pool[best][0], pool[best][1])
    if scattered:
        return ParseResult(scattered, "regex")
    quads = [[float(g) for g in m.groups()] for m in _QUAD_RE.finditer(cleaned)]
    if quads:
        return ParseResult(quads, "regex")
    return ParseResult([], "none")
