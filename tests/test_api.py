import json
import os

import pytest

os.environ.setdefault("VLLMLITE_LAYERS", "2")
os.environ.setdefault("VLLMLITE_DMODEL", "64")
os.environ.setdefault("VLLMLITE_DFF", "128")
os.environ.setdefault("VLLMLITE_HEADS", "4")
os.environ.setdefault("VLLMLITE_KV_HEADS", "2")
os.environ.setdefault("VLLMLITE_NUM_BLOCKS", "256")

from fastapi.testclient import TestClient  # noqa: E402

from vllmlite.api.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_models_lists_something(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list" and body["data"]


def test_completion_shape(client):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 8
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 8


def test_chat_completion_shape(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 6},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] == 6


def test_streaming_deltas_join_to_the_same_text(client):
    payload = {"prompt": "streaming", "max_tokens": 12, "temperature": 0.0, "seed": 3}
    whole = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]

    chunks = []
    with client.stream("POST", "/v1/completions", json={**payload, "stream": True}) as s:
        for line in s.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunks.append(json.loads(body)["choices"][0]["text"])
    assert "".join(chunks) == whole


def test_stream_terminates_with_done(client):
    lines = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 4, "stream": True},
    ) as s:
        lines = [line for line in s.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"


def test_empty_messages_is_a_400(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_bad_sampling_params_do_not_500(client):
    r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 4, "top_p": 5})
    assert r.status_code == 400


def test_metrics_expose_the_cache(client):
    client.post("/v1/completions", json={"prompt": "metrics", "max_tokens": 4})
    m = client.get("/metrics").json()
    for key in ("blocks_total", "blocks_used", "prefix_hit_rate", "preemptions", "steps"):
        assert key in m


def test_concurrent_requests_share_one_engine(client):
    from concurrent.futures import ThreadPoolExecutor

    def go(i):
        return client.post(
            "/v1/completions",
            json={"prompt": f"request number {i}", "max_tokens": 10, "temperature": 0.0},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        rs = list(pool.map(go, range(8)))
    assert all(r.status_code == 200 for r in rs)
    assert all(r.json()["usage"]["completion_tokens"] == 10 for r in rs)
    # every request got its own answer back, not somebody else's
    texts = [r.json()["choices"][0]["text"] for r in rs]
    assert len(set(texts)) > 1
