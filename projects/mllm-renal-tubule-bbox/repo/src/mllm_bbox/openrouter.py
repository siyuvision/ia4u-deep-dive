"""Minimal OpenRouter chat-completions client with retries and a key-free call record."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
KEY_NAME = "OPENROUTER_API_KEY"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class MissingApiKey(RuntimeError):
    pass


def load_api_key(env_file: Path | None = None, environ: dict | None = None) -> str:
    """Environment variable first, then KEY=VALUE lines in env_file."""
    environ = os.environ if environ is None else environ
    value = (environ.get(KEY_NAME) or "").strip()
    if not value and env_file is not None and Path(env_file).is_file():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, raw = line.partition("=")
            if name.strip() == KEY_NAME:
                value = raw.strip().strip("'\"")
    if not value:
        where = f" or in {env_file}" if env_file else ""
        raise MissingApiKey(f"{KEY_NAME} is not set in the environment{where}")
    return value


def build_payload(
    model_id: str,
    prompt: str,
    image_bytes: bytes,
    mime: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> dict:
    data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    payload: dict = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def redact_payload(payload: dict) -> dict:
    """Copy of the payload with the inline image replaced by its size and hash, for storage."""
    redacted = {k: v for k, v in payload.items() if k != "messages"}
    messages = []
    for message in payload["messages"]:
        content = []
        for part in message["content"]:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                b64 = url.split(",", 1)[1]
                raw = base64.b64decode(b64)
                part = {
                    "type": "image_url",
                    "image_url": {"url": f"<inline image {len(raw)} bytes sha256={hashlib.sha256(raw).hexdigest()}>"},
                }
            content.append(part)
        messages.append({**message, "content": content})
    redacted["messages"] = messages
    return redacted


def embedded_error(body) -> dict | None:
    """A provider error carried inside an HTTP 200 body, or None.

    Two shapes occur: a top-level `error` with no `choices` (the request was rejected), and `choices[0].error`
    with finish_reason "error" (the provider dropped the stream part-way, e.g. 502 "Network connection lost").
    Neither is an answer from the model, so neither may be scored as an empty prediction.
    """
    if not isinstance(body, dict):
        return None
    if body.get("error") and not body.get("choices"):
        return body["error"]
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(choice, dict) and (choice.get("error") or choice.get("finish_reason") == "error"):
        return choice.get("error") or {"message": "finish_reason=error"}
    return None


def _retry_delay(attempt: int, response) -> float:
    header = getattr(response, "headers", {}).get("Retry-After") if response is not None else None
    try:
        if header:
            return min(float(header), 60.0)
    except ValueError:
        pass
    return min(4.0 * 2 ** (attempt - 1), 60.0)


def call_chat(
    api_key: str,
    payload: dict,
    *,
    post=requests.post,
    sleep=time.sleep,
    max_attempts: int = 4,
    timeout: tuple[float, float] = (30.0, 900.0),
) -> dict:
    """POST one chat completion. The returned record never contains the API key."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "ia4u mllm-renal-tubule-bbox",
    }
    attempts: list[dict] = []
    started = time.monotonic()
    body = None
    body_text = None
    status: int | None = None
    ok = False
    for attempt in range(1, max_attempts + 1):
        t0 = time.monotonic()
        response = None
        try:
            response = post(API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            status, body, body_text = None, None, None
            attempts.append({"status": None, "error": f"{type(exc).__name__}: {exc}", "latency_s": time.monotonic() - t0})
            retryable = True
        else:
            status = response.status_code
            try:
                body = response.json()
                body_text = None
            except ValueError:
                body, body_text = None, response.text[:2000]
            error = embedded_error(body)
            ok = status == 200 and error is None and isinstance(body, dict)
            attempts.append({"status": status, "error": error, "latency_s": time.monotonic() - t0})
            retryable = (status in RETRY_STATUS) or (status == 200 and not ok)
        if ok or not retryable or attempt == max_attempts:
            break
        sleep(_retry_delay(attempt, response))
    return {
        "ok": ok,
        "status": status,
        "body": body,
        "body_text": body_text,
        "attempts": attempts,
        "latency_s": time.monotonic() - started,
    }


def answer_latency_s(call: dict) -> float | None:
    """Seconds the request that produced the answer took, without earlier failed attempts or backoff sleeps."""
    if not call.get("ok"):
        return None
    attempts = call.get("attempts") or []
    if attempts and isinstance(attempts[-1].get("latency_s"), (int, float)):
        return float(attempts[-1]["latency_s"])
    return call.get("latency_s")


def fetch_generation_stats(
    api_key: str,
    generation_id: str | None,
    *,
    get=requests.get,
    sleep=time.sleep,
    tries: int = 6,
    delay: float = 2.0,
    timeout: float = 30.0,
) -> dict | None:
    """OpenRouter's own record of one generation (time to first token, generation time, native token counts).

    The record shows up a little after the completion, so a 404 is retried. Best effort: any failure returns
    None, because speed statistics must never break a run. The returned dict is OpenRouter's `data` object.
    """
    if not generation_id:
        return None
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(1, tries + 1):
        try:
            response = get(GENERATION_URL, headers=headers, params={"id": generation_id}, timeout=timeout)
        except requests.RequestException:
            response = None
        if response is not None and response.status_code == 200:
            try:
                data = response.json().get("data")
            except (ValueError, AttributeError):
                return None
            return data if isinstance(data, dict) else None
        if response is not None and response.status_code not in (404, 408, 429, 500, 502, 503, 504):
            return None
        if attempt < tries:
            sleep(delay)
    return None


def extract_text(body: dict | None) -> tuple[str, str | None]:
    """Assistant text and finish_reason from a chat-completions body. Reasoning text is never used."""
    try:
        choice = body["choices"][0]
        content = choice["message"].get("content")
        finish = choice.get("finish_reason")
    except (TypeError, KeyError, IndexError):
        return "", None
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or "", finish


def usage_summary(body: dict | None) -> dict:
    body = body if isinstance(body, dict) else {}
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "provider": body.get("provider"),
        "served_model": body.get("model"),
        "generation_id": body.get("id"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "cost_usd": usage.get("cost"),
    }
