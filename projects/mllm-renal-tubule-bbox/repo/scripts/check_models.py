"""Verify the configured model ids against the public OpenRouter model list and snapshot them.

No API key is needed: GET /api/v1/models and GET /api/v1/models/<id>/endpoints are public. The
snapshot records pricing, modalities and supported parameters for the candidate models, plus each
provider endpoint's published price, uptime, latency and throughput (null when OpenRouter has no
recent traffic statistics), so a later reader can see what was available.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "src"))
from mllm_bbox.openrouter import MODELS_URL  # noqa: E402


ENDPOINT_FIELDS = (
    "provider_name",
    "quantization",
    "max_completion_tokens",
    "pricing",
    "uptime_last_30m",
    "latency_last_30m",
    "throughput_last_30m",
)


def fetch_endpoints(model_id: str) -> list[dict] | None:
    """Per-provider endpoint records, or None if the lookup fails (the snapshot then omits them)."""
    try:
        response = requests.get(f"{MODELS_URL}/{model_id}/endpoints", timeout=60)
        response.raise_for_status()
        endpoints = response.json()["data"]["endpoints"]
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None
    return [{k: e.get(k) for k in ENDPOINT_FIELDS} for e in endpoints]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "config" / "models.json")
    parser.add_argument("--out", type=Path, default=None, help="snapshot path; default sources/openrouter-models-<today>.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))["models"]
    listing = requests.get(MODELS_URL, timeout=60)
    listing.raise_for_status()
    catalog = {m["id"]: m for m in listing.json()["data"]}

    snapshot = {"fetched": date.today().isoformat(), "source": MODELS_URL, "models": []}
    problems = 0
    for entry in config:
        model = catalog.get(entry["id"])
        if model is None:
            print(f"MISSING  {entry['key']:<22} {entry['id']}")
            problems += 1
            continue
        modalities = model["architecture"]["input_modalities"]
        has_image = "image" in modalities
        pricing = model["pricing"]
        flag = "" if has_image else "  NO IMAGE INPUT"
        problems += 0 if has_image else 1
        state = "on " if entry["enabled"] else "off"
        print(
            f"{state} {entry['key']:<22} {entry['id']:<34} in={','.join(modalities):<28} "
            f"${float(pricing['prompt']) * 1e6:.3f}/${float(pricing['completion']) * 1e6:.3f} per M{flag}"
        )
        snapshot["models"].append(
            {
                "key": entry["key"],
                "enabled": entry["enabled"],
                "id": model["id"],
                "name": model["name"],
                "input_modalities": modalities,
                "context_length": model.get("context_length"),
                "max_completion_tokens": (model.get("top_provider") or {}).get("max_completion_tokens"),
                "pricing_usd_per_token": {"prompt": pricing["prompt"], "completion": pricing["completion"]},
                "supported_parameters": model.get("supported_parameters"),
                "created": model.get("created"),
                "endpoints": fetch_endpoints(model["id"]),
            }
        )
    out = args.out or PROJECT / "sources" / f"openrouter-models-{snapshot['fetched'].replace('-', '')}.json"
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"snapshot -> {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
