"""Send the image and prompt to each enabled model through OpenRouter and store the raw answers.

Re-running with the same --run-id resumes: a call that already succeeded is not repeated (and not
paid for again). Use --force to repeat everything. Use --dry-run to see the plan and worst-case cost
without needing an API key.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import mimetypes
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "src"))
from mllm_bbox import openrouter as orr  # noqa: E402

DEFAULT_IMAGE = PROJECT / "data" / "input" / "kidney-he-case01-1024x791.jpg"
DEFAULT_PROMPT = REPO / "prompts" / "bbox_v1.txt"
DEFAULT_PRICING = PROJECT / "sources" / "openrouter-models-20261007.json"
EST_INPUT_TOKENS = 3000  # rough image + prompt size, only used for the dry-run cost bound


def load_prompts(models: list[dict], default_path: Path) -> dict[str, tuple[Path, str]]:
    """The prompt each model gets: its own `prompt` file (relative to repo/) when the config names one, else the default."""
    prompts: dict[str, tuple[Path, str]] = {}
    for m in models:
        path = REPO / m["prompt"] if m.get("prompt") else default_path
        prompts[m["key"]] = (path, path.read_text(encoding="utf-8"))
    return prompts


def raw_path(run_dir: Path, key: str, repeat: int) -> Path:
    return run_dir / "raw" / f"{key}__r{repeat}.json"


def already_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        call = json.loads(path.read_text(encoding="utf-8"))["call"]
    except (ValueError, KeyError):
        return False
    # Judged by the current definition of a failed call, so older records with a mid-stream provider error are redone.
    return bool(call["ok"]) and orr.embedded_error(call.get("body")) is None


def attach_generation_stats(run_dir: Path, api_key: str, workers: int) -> tuple[int, int]:
    """Add OpenRouter's /generation record to every answered raw file that lacks one. Best effort.

    Done after the calls rather than inside them: the record appears some time after the completion, and
    waiting for it would hold a worker. Re-running a run also backfills older files. Returns (added, still missing).
    """
    pending = []
    for path in sorted((run_dir / "raw").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["call"]["ok"] and rec.get("generation") is None and rec["usage"].get("generation_id"):
            pending.append((path, rec))

    def fetch(item):
        path, rec = item
        data = orr.fetch_generation_stats(api_key, rec["usage"]["generation_id"])
        if data is None:
            return False
        rec["generation"] = data
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(fetch, pending))
    return sum(results), len(results) - sum(results)


def worst_case_cost(model_id: str, repeats: int, max_tokens: int, pricing_file: Path) -> float | None:
    if not pricing_file.is_file():
        return None
    for entry in json.loads(pricing_file.read_text(encoding="utf-8"))["models"]:
        if entry["id"] == model_id:
            p = entry["pricing_usd_per_token"]
            return repeats * (float(p["prompt"]) * EST_INPUT_TOKENS + float(p["completion"]) * max_tokens)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="comma-separated model keys; default = all enabled in config/models.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "models.json")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help="prompt for every model that does not name its own `prompt` in the config")
    parser.add_argument("--repeats", type=int, default=3, help="independent calls per model")
    parser.add_argument("--max-tokens", type=int, default=32000, help="completion cap, includes reasoning tokens")
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--workers", type=int, default=4, help="concurrent calls; jobs are queued repeat by repeat so concurrent calls are different models")
    parser.add_argument("--no-generation-stats", action="store_true", help="skip the per-call GET /generation lookup (time to first token)")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    args = parser.parse_args(argv)

    models = json.loads(args.config.read_text(encoding="utf-8"))["models"]
    if args.models:
        wanted = [k.strip() for k in args.models.split(",") if k.strip()]
        by_key = {m["key"]: m for m in models}
        unknown = [k for k in wanted if k not in by_key]
        if unknown:
            print(f"unknown model keys: {unknown}; known: {sorted(by_key)}", file=sys.stderr)
            return 2
        selected = [by_key[k] for k in wanted]
    else:
        selected = [m for m in models if m["enabled"]]

    image_bytes = args.image.read_bytes()
    mime = mimetypes.guess_type(args.image.name)[0] or "image/jpeg"
    prompts = load_prompts(selected, args.prompt)
    run_dir = args.results_dir / args.run_id
    # Repeat-major order: the first calls in flight are one per model, so a model does not queue behind itself.
    jobs = [(m, r) for r in range(1, args.repeats + 1) for m in selected]

    print(f"run {args.run_id}: {len(selected)} models x {args.repeats} repeats = {len(jobs)} calls, max_tokens={args.max_tokens}")
    total = 0.0
    for m in selected:
        bound = worst_case_cost(m["id"], args.repeats, args.max_tokens, args.pricing)
        total += bound or 0.0
        shown = f"${bound:.2f}" if bound is not None else "n/a"
        print(f"  {m['key']:<22} {m['id']:<34} worst-case {shown}")
    print(f"  worst-case total (every call uses the full max_tokens): ${total:.2f}")
    if args.dry_run:
        return 0

    try:
        api_key = orr.load_api_key(REPO / ".env")
    except orr.MissingApiKey as exc:
        print(f"{exc}\nPut the key in {REPO / '.env'} as OPENROUTER_API_KEY=... (copy .env.example).", file=sys.stderr)
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": args.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "models": selected,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "image": {"path": str(args.image), "sha256": hashlib.sha256(image_bytes).hexdigest(), "bytes": len(image_bytes)},
        "prompts": {key: {"path": str(path), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()} for key, (path, text) in prompts.items()},
        "python": platform.python_version(),
        "requests": requests.__version__,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def run_one(model: dict, repeat: int) -> str:
        path = raw_path(run_dir, model["key"], repeat)
        if not args.force and already_done(path):
            return f"skip   {model['key']} r{repeat} (already done)"
        payload = orr.build_payload(model["id"], prompts[model["key"]][1], image_bytes, mime, args.max_tokens, args.reasoning_effort)
        call = orr.call_chat(api_key, payload)
        text, finish = orr.extract_text(call["body"])
        usage = orr.usage_summary(call["body"])
        record = {
            "model": model,
            "repeat": repeat,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "request": orr.redact_payload(payload),
            "call": call,
            "text": text,
            "finish_reason": finish,
            "usage": usage,
            "generation": None,  # filled by the pass after all calls, once OpenRouter has written its own records
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        status = "ok  " if call["ok"] else "FAIL"
        cost = usage["cost_usd"]
        seconds = orr.answer_latency_s(call) if call["ok"] else call["latency_s"]
        return (
            f"{status}   {model['key']} r{repeat} status={call['status']} {seconds:.1f}s "
            f"chars={len(text)} finish={finish} cost={'n/a' if cost is None else f'${cost:.4f}'}"
        )

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, m, r): (m, r) for m, r in jobs}
        for future in concurrent.futures.as_completed(futures):
            model, repeat = futures[future]
            try:
                line = future.result()
            except Exception as exc:  # keep the rest of the run going; the failed call is retried on resume
                line = f"FAIL   {model['key']} r{repeat} exception {type(exc).__name__}: {exc}"
            failures += line.startswith("FAIL")
            print(line, flush=True)
    print(f"done: raw answers in {run_dir / 'raw'}; {failures} failed call(s)")

    if not args.no_generation_stats:
        filled, missing = attach_generation_stats(run_dir, api_key, args.workers)
        print(f"generation stats (time to first token): {filled} added, {missing} unavailable")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
