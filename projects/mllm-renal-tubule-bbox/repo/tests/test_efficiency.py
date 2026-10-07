import pytest
import requests

from mllm_bbox import openrouter as orr
from mllm_bbox.efficiency import cost_per_true_positive, list_price_per_million, non_dominated, speed_fields


def record(ok=True, attempts=None, wall=9.0, completion=300, generation=None):
    return {
        "call": {"ok": ok, "attempts": attempts if attempts is not None else [{"status": 200, "latency_s": 3.0}], "latency_s": wall},
        "usage": {"completion_tokens": completion},
        "generation": generation,
    }


def test_latency_is_the_answering_request_not_the_retries():
    row = speed_fields(record(attempts=[{"status": 429, "latency_s": 1.0}, {"status": 200, "latency_s": 3.0}], wall=9.0))
    assert row["latency_s"] == 3.0 and row["wall_s"] == 9.0 and row["attempts"] == 2
    assert row["tok_per_s"] == pytest.approx(100.0)  # 300 tokens / 3 s
    assert row["ttft_s"] is None and row["gen_s"] is None


def test_generation_record_milliseconds_become_seconds():
    row = speed_fields(record(generation={"latency": 1500, "generation_time": 2500}))
    assert row["ttft_s"] == pytest.approx(1.5) and row["gen_s"] == pytest.approx(2.5)


def test_failed_call_has_no_speed_and_missing_token_count_has_no_rate():
    failed = speed_fields(record(ok=False))
    assert failed["latency_s"] is None and failed["tok_per_s"] is None
    assert speed_fields(record(completion=None))["tok_per_s"] is None


def test_cost_per_true_positive_is_mean_cost_over_mean_matches():
    assert cost_per_true_positive([0.02, 0.04], [10, 20]) == pytest.approx(0.03 / 15)
    assert cost_per_true_positive([0.02], [0]) is None  # nothing matched: no price per match
    assert cost_per_true_positive([], [5]) is None


def test_list_price_is_per_million_tokens():
    entry = {"pricing_usd_per_token": {"prompt": "0.00000015", "completion": "0.00000047"}}
    assert list_price_per_million(entry) == pytest.approx((0.15, 0.47))
    assert list_price_per_million(None) is None
    assert list_price_per_million({"pricing_usd_per_token": {}}) is None


def test_non_dominated_keeps_the_quality_cost_frontier():
    points = {
        "cheap-weak": (0.2, 0.001),
        "mid": (0.6, 0.01),
        "best": (0.9, 0.05),
        "wasteful": (0.5, 0.05),  # beaten by mid: lower quality, higher cost
        "twin-of-mid": (0.6, 0.01),  # exact tie: neither dominates
    }
    assert sorted(non_dominated(points)) == ["best", "cheap-weak", "mid", "twin-of-mid"]


def test_non_dominated_same_quality_cheaper_wins():
    assert non_dominated({"a": (0.5, 0.02), "b": (0.5, 0.01)}) == ["b"]


class Seq:
    def __init__(self, *items):
        self.items = list(items)
        self.urls = []

    def __call__(self, url, headers=None, params=None, timeout=None):
        self.urls.append((url, params))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def test_generation_stats_waits_for_the_record_to_appear():
    get = Seq(Resp(404), Resp(404), Resp(200, {"data": {"latency": 700, "generation_time": 900}}))
    sleeps = []
    data = orr.fetch_generation_stats("sk-or-x", "gen-9", get=get, sleep=sleeps.append, delay=2.0)
    assert data == {"latency": 700, "generation_time": 900}
    assert sleeps == [2.0, 2.0] and get.urls[0] == (orr.GENERATION_URL, {"id": "gen-9"})


def test_generation_stats_failures_never_raise():
    assert orr.fetch_generation_stats("k", None, get=Seq()) is None
    assert orr.fetch_generation_stats("k", "g", get=Seq(Resp(401)), sleep=lambda s: pytest.fail("401 is final")) is None
    assert orr.fetch_generation_stats("k", "g", get=Seq(Resp(404), Resp(404)), sleep=lambda s: None, tries=2) is None
    assert orr.fetch_generation_stats("k", "g", get=Seq(requests.ConnectionError("x"), Resp(200, {"data": {"latency": 1}})), sleep=lambda s: None) == {"latency": 1}
    assert orr.fetch_generation_stats("k", "g", get=Seq(Resp(200)), sleep=lambda s: None) is None  # 200 but not JSON
    assert orr.fetch_generation_stats("k", "g", get=Seq(Resp(200, {"data": None})), sleep=lambda s: None) is None
