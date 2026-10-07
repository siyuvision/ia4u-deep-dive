import base64
import hashlib
import json

import pytest
import requests

from mllm_bbox import openrouter as orr

KEY = "sk-or-test-SECRET-123"


class FakeResponse:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code = status
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def ok_body(content="[]"):
    return {"id": "gen-1", "model": "x/y", "provider": "P", "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


class Sequence:
    """post() stand-in returning or raising the given items in order."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls = 0

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_load_api_key_prefers_environment_then_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nOPENROUTER_API_KEY='file-key'\n", encoding="utf-8")
    assert orr.load_api_key(env, environ={"OPENROUTER_API_KEY": "env-key"}) == "env-key"
    assert orr.load_api_key(env, environ={}) == "file-key"


def test_load_api_key_missing_or_blank(tmp_path):
    blank = tmp_path / ".env"
    blank.write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")
    with pytest.raises(orr.MissingApiKey):
        orr.load_api_key(blank, environ={})
    with pytest.raises(orr.MissingApiKey):
        orr.load_api_key(tmp_path / "absent.env", environ={})


def test_payload_carries_image_and_redaction_removes_it():
    image = b"\xff\xd8\xff-fake-jpeg-bytes"
    payload = orr.build_payload("a/b", "prompt text", image, "image/jpeg", 1234, reasoning_effort="low")
    parts = payload["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "prompt text"}
    url = parts[1]["image_url"]["url"]
    assert url == "data:image/jpeg;base64," + base64.b64encode(image).decode()
    assert payload["model"] == "a/b" and payload["max_tokens"] == 1234
    assert payload["reasoning"] == {"effort": "low"} and payload["usage"] == {"include": True}
    assert "temperature" not in payload  # provider defaults are part of the design
    red = orr.redact_payload(payload)
    assert base64.b64encode(image).decode() not in json.dumps(red)
    assert hashlib.sha256(image).hexdigest() in json.dumps(red)
    assert "reasoning" not in orr.build_payload("a/b", "p", image, "image/jpeg", 10)


def test_retries_rate_limit_then_succeeds_and_honours_retry_after():
    post = Sequence(FakeResponse(429, {"error": {"message": "slow down"}}, headers={"Retry-After": "7"}), FakeResponse(200, ok_body()))
    sleeps = []
    rec = orr.call_chat(KEY, {"model": "a/b"}, post=post, sleep=sleeps.append)
    assert rec["ok"] and rec["status"] == 200 and post.calls == 2
    assert sleeps == [7.0]
    assert [a["status"] for a in rec["attempts"]] == [429, 200]


def test_http_200_with_embedded_provider_error_is_retried_then_reported():
    err = FakeResponse(200, {"error": {"code": 502, "message": "upstream"}})
    post = Sequence(err, err, err)
    rec = orr.call_chat(KEY, {}, post=post, sleep=lambda s: None, max_attempts=3)
    assert not rec["ok"] and post.calls == 3
    assert rec["attempts"][-1]["error"]["message"] == "upstream"


DROPPED_STREAM = {
    "id": "gen-2",
    "provider": "P",
    "choices": [
        {
            "finish_reason": "error",
            "error": {"code": 502, "message": "Network connection lost."},
            "message": {"role": "assistant", "content": None, "reasoning": "partial thinking..."},
        }
    ],
    "usage": {"completion_tokens": 11677, "cost": 0},
}


def test_stream_dropped_midway_is_a_failed_call_not_an_empty_answer():
    assert orr.embedded_error(DROPPED_STREAM)["code"] == 502
    assert orr.embedded_error({"choices": [{"finish_reason": "error", "message": {"content": None}}]}) is not None
    assert orr.embedded_error({"error": {"message": "rejected"}}) is not None
    assert orr.embedded_error(ok_body("[]")) is None and orr.embedded_error(None) is None
    # An empty answer that ended normally is the model's answer and stays scoreable.
    assert orr.embedded_error({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}) is None
    post = Sequence(FakeResponse(200, DROPPED_STREAM), FakeResponse(200, ok_body("[]")))
    rec = orr.call_chat(KEY, {}, post=post, sleep=lambda s: None)
    assert rec["ok"] and post.calls == 2  # retried, and the retry's answer is the one kept
    assert rec["attempts"][0]["error"]["message"] == "Network connection lost."
    dead = orr.call_chat(KEY, {}, post=Sequence(*[FakeResponse(200, DROPPED_STREAM)] * 2), sleep=lambda s: None, max_attempts=2)
    assert not dead["ok"] and dead["status"] == 200


def test_client_error_is_not_retried_and_body_is_kept():
    post = Sequence(FakeResponse(400, {"error": {"message": "image not supported"}}))
    rec = orr.call_chat(KEY, {}, post=post, sleep=lambda s: pytest.fail("must not sleep"))
    assert not rec["ok"] and rec["status"] == 400 and post.calls == 1
    assert rec["body"]["error"]["message"] == "image not supported"


def test_network_error_is_retried():
    post = Sequence(requests.ConnectionError("boom"), FakeResponse(200, ok_body()))
    rec = orr.call_chat(KEY, {}, post=post, sleep=lambda s: None)
    assert rec["ok"] and rec["attempts"][0]["status"] is None and "ConnectionError" in rec["attempts"][0]["error"]


def test_non_json_body_is_stored_as_text():
    rec = orr.call_chat(KEY, {}, post=Sequence(FakeResponse(401, None, text="Unauthorized")), sleep=lambda s: None)
    assert not rec["ok"] and rec["body"] is None and rec["body_text"] == "Unauthorized"


def test_call_record_never_contains_the_api_key():
    rec = orr.call_chat(KEY, {"model": "a/b"}, post=Sequence(FakeResponse(200, ok_body())), sleep=lambda s: None)
    assert KEY not in json.dumps(rec)


def test_extract_text_and_usage():
    assert orr.extract_text(ok_body("hello")) == ("hello", "stop")
    parts = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}, "finish_reason": "length"}]}
    assert orr.extract_text(parts) == ("ab", "length")
    assert orr.extract_text({"choices": [{"message": {"content": None}, "finish_reason": "length"}]}) == ("", "length")
    assert orr.extract_text(None) == ("", None)
    assert orr.extract_text({"error": {}}) == ("", None)
    body = ok_body()
    body["usage"] = {"prompt_tokens": 10, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 40}, "cost": 0.0123}
    u = orr.usage_summary(body)
    assert (u["provider"], u["prompt_tokens"], u["completion_tokens"], u["reasoning_tokens"], u["cost_usd"]) == ("P", 10, 50, 40, 0.0123)
    assert orr.usage_summary(None)["cost_usd"] is None
