"""End to end: run_models (network faked) -> raw answers -> evaluate_runs, on a tiny synthetic case."""

import csv
import json

import pytest
from PIL import Image

from conftest import load_script
from mllm_bbox import openrouter as orr
from mllm_bbox.boxes import Box, boxes_from_center_json, raw_to_box
from mllm_bbox.evaluation import evaluate_text

SENT = (100, 80)
THRESHOLDS = (0.3, 0.5, 0.75)


def center(box: Box):
    return {"x": (box.x1 + box.x2) / 2, "y": (box.y1 + box.y2) / 2, "width": box.w, "height": box.h, "class": "renal tubule"}


# Non-square boxes so that swapping x and y changes them. Raw values are xyxy_norm1000.
RAW = [[100, 200, 300, 500], [500, 100, 800, 300]]
GT_BOXES = [raw_to_box(r, "xyxy_norm1000", SENT) for r in RAW]
ANSWER_OK = json.dumps([{"label": "renal tubule", "bbox_2d": r} for r in RAW])
ANSWER_YX = json.dumps([{"label": "renal tubule", "bbox_2d": [r[1], r[0], r[3], r[2]]} for r in RAW])


def test_perfect_answer_scores_one():
    row, pred, _ = evaluate_text(ANSWER_OK, GT_BOXES, SENT, "xyxy_norm1000", THRESHOLDS, 10)
    assert row["f1@0.5"] == pytest.approx(1.0) and row["f1@0.75"] == pytest.approx(1.0)
    assert row["parse_mode"] == "json" and row["n_pred"] == 2 and not row["convention_mismatch"]
    assert row["mean_matched_iou"] == pytest.approx(1.0)
    assert row["f1_any_convention@0.5"] == pytest.approx(1.0)


def test_swapped_axis_answer_is_flagged_as_convention_mismatch():
    row, _, _ = evaluate_text(ANSWER_YX, GT_BOXES, SENT, "xyxy_norm1000", THRESHOLDS, 10)
    assert row["f1@0.5"] < 0.5
    assert row["convention_mismatch"] and row["best_alt_convention"] == "yxyx_norm1000"
    assert row["best_alt_f1"] == pytest.approx(1.0)
    assert row["f1_any_convention@0.5"] == pytest.approx(1.0)  # the locations are right, only the axis order differs


def test_unusable_answer_scores_zero_without_crashing():
    row, pred, _ = evaluate_text("I cannot do that.", GT_BOXES, SENT, "xyxy_norm1000", THRESHOLDS, 10)
    assert row["parse_mode"] == "none" and pred == [] and row["f1@0.5"] == 0.0


def test_run_then_evaluate(tmp_path, monkeypatch):
    image = tmp_path / "img.jpg"
    Image.new("RGB", SENT, (200, 120, 160)).save(image)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"image": {"width": 1164, "height": 900}, "predictions": [center(b) for b in GT_BOXES]}), encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("find tubules", encoding="utf-8")
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "models": [
                    {"key": "good", "label": "Good model", "id": "x/good", "enabled": True},
                    {"key": "swapped", "label": "Swapped model", "id": "x/swapped", "enabled": True},
                    {"key": "off", "label": "Disabled", "id": "x/off", "enabled": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_call(api_key, payload, **kwargs):
        calls.append(payload["model"])
        text = ANSWER_OK if payload["model"] == "x/good" else ANSWER_YX
        body = {
            "id": "gen-1",
            "model": payload["model"],
            "provider": "FakeProvider",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
        }
        # The first attempt failed and was retried: latency_s is the final request, wall time includes the retry.
        attempts = [{"status": 502, "latency_s": 0.3}, {"status": 200, "latency_s": 0.4}]
        return {"ok": True, "status": 200, "body": body, "body_text": None, "attempts": attempts, "latency_s": 6.7}

    def fake_generation(api_key, generation_id, **kwargs):
        assert generation_id == "gen-1"
        return {"latency": 400, "generation_time": 1200}  # milliseconds, as in OpenRouter's /generation record

    monkeypatch.setattr(orr, "call_chat", fake_call)
    monkeypatch.setattr(orr, "fetch_generation_stats", fake_generation)
    monkeypatch.setattr(orr, "load_api_key", lambda *a, **k: "sk-or-fake-key")
    run_models = load_script("run_models")
    evaluate_runs = load_script("evaluate_runs")
    results = tmp_path / "results"
    common = ["--results-dir", str(results), "--run-id", "t1"]

    assert run_models.main(common + ["--config", str(config), "--image", str(image), "--prompt", str(prompt), "--repeats", "2", "--workers", "1"]) == 0
    assert sorted(calls) == ["x/good", "x/good", "x/swapped", "x/swapped"]  # disabled model never called
    assert calls[:2] == ["x/good", "x/swapped"]  # repeat-major: concurrent calls are different models
    raw = json.loads((results / "t1" / "raw" / "good__r1.json").read_text(encoding="utf-8"))
    assert "sk-or-fake-key" not in json.dumps(raw)
    assert "base64," not in json.dumps(raw["request"])

    # Resuming does not repeat calls that already succeeded.
    assert run_models.main(common + ["--config", str(config), "--image", str(image), "--prompt", str(prompt), "--repeats", "2", "--workers", "1"]) == 0
    assert len(calls) == 4

    assert evaluate_runs.main(common + ["--image", str(image), "--gt", str(gt)]) == 0
    rows = list(csv.DictReader((results / "t1" / "scores.csv").open(encoding="utf-8")))
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(float(r["f1@0.5"]))
    assert by_model["good"] == [pytest.approx(1.0)] * 2
    assert all(v < 0.5 for v in by_model["swapped"])
    summary = (results / "t1" / "summary.md").read_text(encoding="utf-8")
    assert "Good model" in summary and "convention? yxyx_norm1000" in summary

    # Speed and price, from the fake call: 20 tokens in 0.4 s (not the 6.7 s wall time) = 50 tok/s; $0.001 per call.
    good = [r for r in rows if r["model"] == "good"][0]
    assert good["latency_s"] == "0.4" and good["wall_s"] == "6.7" and good["attempts"] == "2"
    assert float(good["tok_per_s"]) == pytest.approx(50.0) and float(good["ttft_s"]) == pytest.approx(0.4) and float(good["gen_s"]) == pytest.approx(1.2)
    assert json.loads((results / "t1" / "raw" / "good__r1.json").read_text(encoding="utf-8"))["generation"] == {"latency": 400, "generation_time": 1200}
    speed_table = summary.split("## Speed and price")[1]
    good_row = next(line for line in speed_table.splitlines() if line.startswith("| Good model"))
    assert "FakeProvider" in good_row and "$0.0010" in good_row and "$0.0020" in good_row  # per call, total of 2 calls
    assert "$0.0005" in good_row  # $0.001 per call / 2 boxes matched
    swapped_row = next(line for line in speed_table.splitlines() if line.startswith("| Swapped model"))
    assert swapped_row.split("|")[-3].strip() == "-"  # nothing matched, so no price per match
    assert "Not beaten on F1 vs $/call" in speed_table and "Good model" in speed_table.split("Not beaten on F1 vs $/call")[1].splitlines()[0]
    assert "Swapped model" not in speed_table.split("Not beaten on F1 vs $/call")[1].splitlines()[0]  # same cost, lower F1
    assert (results / "t1" / "overlays" / "good__r1.png").is_file()
    parsed = boxes_from_center_json(json.loads((results / "t1" / "parsed" / "good__r1.json").read_text(encoding="utf-8")))
    assert len(parsed) == 2


def test_model_with_its_own_prompt_and_convention(tmp_path, monkeypatch):
    """A model whose config names a prompt and a convention is asked and scored in that format; the others keep the defaults."""
    image = tmp_path / "img.jpg"
    Image.new("RGB", SENT).save(image)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"image": {"width": 1164, "height": 900}, "predictions": [center(b) for b in GT_BOXES]}), encoding="utf-8")
    default_prompt = tmp_path / "default.txt"
    default_prompt.write_text("default prompt", encoding="utf-8")
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "models": [
                    {"key": "plain", "label": "Plain", "id": "x/plain", "enabled": True},
                    {"key": "yx", "label": "Own format", "id": "x/yx", "enabled": True, "prompt": "prompts/bbox_v1_gemini.txt", "convention": "yxyx_norm1000"},
                ]
            }
        ),
        encoding="utf-8",
    )
    seen = {}

    def fake_call(api_key, payload, **kwargs):
        seen[payload["model"]] = payload["messages"][0]["content"][0]["text"]
        text = ANSWER_OK if payload["model"] == "x/plain" else ANSWER_YX  # x/yx answers the way its prompt asked
        body = {"id": "g", "choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"cost": 0.0}}
        return {"ok": True, "status": 200, "body": body, "body_text": None, "attempts": [{"status": 200, "latency_s": 1.0}], "latency_s": 1.0}

    monkeypatch.setattr(orr, "call_chat", fake_call)
    monkeypatch.setattr(orr, "fetch_generation_stats", lambda *a, **k: None)
    monkeypatch.setattr(orr, "load_api_key", lambda *a, **k: "sk-or-fake-key")
    run_models = load_script("run_models")
    evaluate_runs = load_script("evaluate_runs")
    results = tmp_path / "results"
    common = ["--results-dir", str(results), "--run-id", "own"]
    assert run_models.main(common + ["--config", str(config), "--image", str(image), "--prompt", str(default_prompt), "--repeats", "1", "--no-generation-stats"]) == 0
    assert seen["x/plain"] == "default prompt"
    assert seen["x/yx"] == (run_models.REPO / "prompts" / "bbox_v1_gemini.txt").read_text(encoding="utf-8")
    prompts = json.loads((results / "own" / "run.json").read_text(encoding="utf-8"))["prompts"]
    assert prompts["plain"]["sha256"] != prompts["yx"]["sha256"]

    assert evaluate_runs.main(common + ["--image", str(image), "--gt", str(gt), "--no-overlays"]) == 0
    rows = {r["model"]: r for r in csv.DictReader((results / "own" / "scores.csv").open(encoding="utf-8"))}
    assert rows["plain"]["convention"] == "xyxy_norm1000" and rows["yx"]["convention"] == "yxyx_norm1000"
    assert float(rows["yx"]["f1@0.5"]) == pytest.approx(1.0)  # scored in the format it was asked for
    assert rows["yx"]["convention_mismatch"] == "False"


def test_native_format_config_points_at_real_prompts_and_known_conventions():
    from mllm_bbox.boxes import CONVENTIONS

    repo = load_script("run_models").REPO
    project = repo.parent
    models = json.loads((repo / "config" / "models-native.json").read_text(encoding="utf-8"))["models"]
    shared_ids = {m["id"] for m in json.loads((repo / "config" / "models.json").read_text(encoding="utf-8"))["models"]}
    with Image.open(project / "data" / "input" / "kidney-he-case01-1024x791.jpg") as im:
        width, height = im.size
    for m in models:
        assert m["convention"] in CONVENTIONS and m["id"] in shared_ids
        text = (repo / m["prompt"]).read_text(encoding="utf-8")
        if m["convention"].endswith("_px"):  # a pixel prompt must state the size of the image that is actually sent
            assert f"{width} pixels wide" in text and f"{height} pixels high" in text


def test_stored_record_with_midstream_error_is_redone_and_not_scored(tmp_path, monkeypatch):
    """A record saved by an older client as ok=True, whose body shows the provider dropped the stream."""
    run_models = load_script("run_models")
    evaluate_runs = load_script("evaluate_runs")
    image = tmp_path / "img.jpg"
    Image.new("RGB", SENT).save(image)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"image": {"width": 1164, "height": 900}, "predictions": [center(b) for b in GT_BOXES]}), encoding="utf-8")
    results = tmp_path / "results"
    path = results / "r" / "raw" / "m__r1.json"
    path.parent.mkdir(parents=True)
    dropped = {"choices": [{"finish_reason": "error", "error": {"code": 502}, "message": {"content": None}}]}
    path.write_text(
        json.dumps(
            {
                "model": {"key": "m", "label": "M", "id": "x/m"},
                "repeat": 1,
                "call": {"ok": True, "status": 200, "body": dropped, "attempts": [{"status": 200, "latency_s": 5.0}], "latency_s": 5.0},
                "text": "",
                "finish_reason": "error",
                "usage": {"cost_usd": 0, "completion_tokens": 100},
            }
        ),
        encoding="utf-8",
    )
    assert not run_models.already_done(path)  # would be called again on resume
    assert evaluate_runs.main(["--results-dir", str(results), "--run-id", "r", "--image", str(image), "--gt", str(gt)]) == 0
    row = list(csv.DictReader((results / "r" / "scores.csv").open(encoding="utf-8")))[0]
    assert row["call_ok"] == "False" and "f1@0.5" not in row  # not an empty prediction scored as 0
    assert "1 call(s) failed" in (results / "r" / "summary.md").read_text(encoding="utf-8")


def test_dry_run_needs_no_key_and_makes_no_calls(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("dry run must not call the API")

    monkeypatch.setattr(orr, "call_chat", boom)
    monkeypatch.setattr(orr, "load_api_key", boom)
    run_models = load_script("run_models")
    image = tmp_path / "img.jpg"
    Image.new("RGB", SENT).save(image)
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"models": [{"key": "m", "label": "M", "id": "x/m", "enabled": True}]}), encoding="utf-8")
    assert run_models.main(["--dry-run", "--config", str(config), "--image", str(image), "--results-dir", str(tmp_path / "r")]) == 0
    assert "1 models x 3 repeats" in capsys.readouterr().out
    assert not (tmp_path / "r").exists()
