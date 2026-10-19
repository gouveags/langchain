"""Request-body compression contracts through the real Anthropic SDK."""

from __future__ import annotations

import gzip
import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic
import pytest
from langchain_core.messages import HumanMessage

from langchain_anthropic import ChatAnthropic, _client_utils
from tests.unit_tests._httpx_compat import httpx

_RESPONSE: dict[str, Any] = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-test",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 2, "output_tokens": 1},
}


@pytest.fixture
async def wire_requests(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[Any]]:
    requests: list[Any] = []
    sync_clients: list[Any] = []
    async_clients: list[Any] = []
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    for variable in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_PROXY",
        "LANGSMITH_GATEWAY",
        "LANGSMITH_GATEWAY_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)
    _client_utils._get_default_httpx_client.cache_clear()
    _client_utils._get_default_async_httpx_client.cache_clear()

    def handler(request: Any) -> Any:
        requests.append(request)
        if request.headers.get("x-redirect-target") and len(requests) == 1:
            return httpx.Response(
                int(request.headers["x-redirect-status"]),
                headers={"Location": request.headers["x-redirect-target"]},
            )
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 9})
        if request.headers.get("x-retry-test") and len(requests) == 1:
            return httpx.Response(
                429,
                headers={"retry-after-ms": "1"},
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "retry"},
                },
            )
        body = (
            gzip.decompress(request.content)
            if request.headers.get("Content-Encoding") == "gzip"
            else request.content
        )
        if json.loads(body).get("stream"):
            events: list[dict[str, Any]] = [
                {
                    "type": "message_start",
                    "message": {**_RESPONSE, "content": [], "stop_reason": None},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
            return httpx.Response(
                200,
                text="".join(
                    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    for event in events
                ),
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json=_RESPONSE)

    original_sync = _client_utils._SyncHttpxClientWrapper.__init__
    original_async = _client_utils._AsyncHttpxClientWrapper.__init__

    def initialize_sync(self: Any, **kwargs: Any) -> None:
        original_sync(
            self,
            **{**kwargs, "transport": httpx.MockTransport(handler), "trust_env": False},
        )
        sync_clients.append(self)

    def initialize_async(self: Any, **kwargs: Any) -> None:
        original_async(
            self,
            **{**kwargs, "transport": httpx.MockTransport(handler), "trust_env": False},
        )
        async_clients.append(self)

    monkeypatch.setattr(
        _client_utils._SyncHttpxClientWrapper, "__init__", initialize_sync
    )
    monkeypatch.setattr(
        _client_utils._AsyncHttpxClientWrapper, "__init__", initialize_async
    )
    yield requests
    for client in sync_clients:
        client.close()
    for client in async_clients:
        await client.aclose()
    _client_utils._get_default_httpx_client.cache_clear()
    _client_utils._get_default_async_httpx_client.cache_clear()


def make_model(**options: Any) -> ChatAnthropic:
    defaults: dict[str, Any] = {
        "model": "claude-test",
        "api_key": "unit-test-key",
        "max_retries": 0,
    }
    return ChatAnthropic(**{**defaults, **options})


def assert_wire(request: Any, compression: str | None) -> dict[str, Any]:
    assert request.url.host == "api.anthropic.com"
    assert request.url.path == "/v1/messages"
    if compression:
        assert request.headers["Content-Encoding"] == "gzip"
        content = gzip.decompress(request.content)
    else:
        assert "Content-Encoding" not in request.headers
        content = request.content
    assert int(request.headers["Content-Length"]) == len(request.content)
    assert request.headers["Content-Type"].startswith("application/json")
    payload: dict[str, Any] = json.loads(content)
    assert "request_compression" not in payload
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    return payload


@pytest.mark.parametrize("compression", [None, "gzip"])
def test_invoke_sends_serialized_json_with_selected_encoding(
    wire_requests: list[Any], compression: str | None
) -> None:
    model = make_model(
        **({} if compression is None else {"request_compression": compression})
    )
    assert model.invoke("hello").content == "hello"
    assert_wire(wire_requests[0], compression)


@pytest.mark.parametrize("compression", [None, "gzip"])
async def test_ainvoke_sends_compressed_json(
    wire_requests: list[Any], compression: str | None
) -> None:
    model = make_model(
        **({} if compression is None else {"request_compression": compression})
    )
    assert (await model.ainvoke("hello")).content == "hello"
    assert_wire(wire_requests[0], compression)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_preserves_sse_response(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    model = make_model(request_compression="gzip")
    chunks = (
        [chunk async for chunk in model.astream("hello")]
        if asynchronous
        else list(model.stream("hello"))
    )
    assert "".join(str(chunk.text) for chunk in chunks) == "hello"
    assert assert_wire(wire_requests[0], "gzip")["stream"] is True


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_sdk_retry_compresses_once_per_attempt(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="gzip",
        max_retries=1,
        default_headers={"x-retry-test": "1"},
    )
    result = await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert result.content == "hello"
    assert len(wire_requests) == 2
    payloads = [assert_wire(request, "gzip") for request in wire_requests]
    assert payloads[0] == payloads[1]
    assert wire_requests[0].content == wire_requests[1].content


@pytest.mark.parametrize("codec", ["zstd", "br", True])
def test_rejects_unsupported_codec(wire_requests: list[Any], codec: Any) -> None:
    with pytest.raises(ValueError, match="request_compression"):
        make_model(request_compression=codec)
    assert wire_requests == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.anthropic.com",
        "https://api.anthropic.com.evil.test",
        "https://api.anthropic.com:444",
        "https://user@api.anthropic.com",
        "https://api.anthropic.com/custom",
        "https://bedrock-mantle.us-east-1.api.aws/anthropic",
    ],
)
def test_rejects_unsupported_destination(
    wire_requests: list[Any], base_url: str
) -> None:
    with pytest.raises(ValueError, match=r"compression|api\.anthropic\.com"):
        make_model(request_compression="gzip", base_url=base_url).invoke("hello")
    assert wire_requests == []


def test_encoding_conflict_fails_before_send(wire_requests: list[Any]) -> None:
    with pytest.raises(ValueError, match=r"Content-Encoding|content-encoding"):
        make_model(
            request_compression="gzip", default_headers={"content-encoding": "br"}
        ).invoke("hello")
    assert wire_requests == []


def test_replaces_length_and_preserves_other_headers_and_timeout(
    wire_requests: list[Any],
) -> None:
    make_model(
        request_compression="gzip",
        timeout=2,
        default_headers={"content-length": "999", "x-custom": "preserved"},
    ).invoke("hello")
    request = wire_requests[0]
    assert_wire(request, "gzip")
    assert request.headers["x-custom"] == "preserved"
    assert request.headers["x-api-key"] == "unit-test-key"
    assert request.extensions["timeout"]["read"] == 2


def test_enabled_and_default_clients_do_not_leak_configuration(
    wire_requests: list[Any],
) -> None:
    enabled = make_model(request_compression="gzip")
    disabled = make_model()
    enabled.invoke("hello")
    disabled.invoke("hello")
    make_model(request_compression="gzip").invoke("hello")
    assert [request.headers.get("Content-Encoding") for request in wire_requests] == [
        "gzip",
        None,
        "gzip",
    ]
    assert enabled._client._client is not disabled._client._client


def test_token_counting_keeps_auxiliary_request_uncompressed(
    wire_requests: list[Any],
) -> None:
    model = make_model(request_compression="gzip")
    assert model.get_num_tokens_from_messages([HumanMessage(content="hello")]) == 9
    assert wire_requests[0].url.path == "/v1/messages/count_tokens"
    assert "Content-Encoding" not in wire_requests[0].headers


@pytest.mark.parametrize(
    ("variable", "value"),
    [("ANTHROPIC_BASE_URL", "https://proxy.example"), ("LANGSMITH_GATEWAY", "true")],
)
def test_environment_destination_is_validated(
    wire_requests: list[Any], monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="compression"):
        make_model(request_compression="gzip").invoke("hello")
    assert wire_requests == []


def test_beta_messages_query_uses_same_compression(wire_requests: list[Any]) -> None:
    model = make_model(request_compression="gzip")
    result = model._client.beta.messages.create(
        model="claude-test",
        max_tokens=16,
        messages=[{"role": "user", "content": "hello"}],
        betas=["test-beta"],
    )
    assert result.content[0].type == "text"
    assert result.content[0].text == "hello"
    assert_wire(wire_requests[0], "gzip")
    assert wire_requests[0].url.params["beta"] == "true"


def test_non_json_inference_request_is_rejected(wire_requests: list[Any]) -> None:
    model = make_model(request_compression="gzip")
    with pytest.raises(ValueError, match="JSON"):
        model._client._client.build_request(
            "POST",
            "https://api.anthropic.com/v1/messages",
            content=b"opaque",
            headers={"Content-Type": "application/octet-stream"},
        )
    assert wire_requests == []


def test_unbuffered_json_request_fails_clearly(wire_requests: list[Any]) -> None:
    model = make_model(request_compression="gzip")
    with pytest.raises(ValueError, match=r"buffered|JSON"):
        model._client._client.build_request(
            "POST",
            "https://api.anthropic.com/v1/messages",
            content=iter([b'{"model":"test"}']),
            headers={"Content-Type": "application/json"},
        )
    assert wire_requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize(
    "target",
    [
        "https://unapproved.example/v1/messages",
        "https://api.anthropic.com/v1/messages/count_tokens",
    ],
)
async def test_redirect_cannot_send_compressed_body_to_unapproved_destination(
    wire_requests: list[Any], status: int, target: str, *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="gzip",
        default_headers={"x-redirect-target": target, "x-redirect-status": str(status)},
    )
    with pytest.raises((ValueError, anthropic.APIConnectionError)):
        await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert len(wire_requests) == 1
    assert_wire(wire_requests[0], "gzip")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [307, 308])
async def test_supported_query_redirect_preserves_single_compressed_body(
    wire_requests: list[Any], status: int, *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="gzip",
        default_headers={
            "x-redirect-target": "https://api.anthropic.com/v1/messages?redirected=true",
            "x-redirect-status": str(status),
        },
    )
    result = await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert result.text == "hello"
    assert len(wire_requests) == 2
    assert_wire(wire_requests[0], "gzip")
    assert_wire(wire_requests[1], "gzip")
    assert wire_requests[0].content == wire_requests[1].content
    assert wire_requests[1].url.params["redirected"] == "true"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_method_rewriting_redirect_is_rejected(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="gzip",
        default_headers={
            "x-redirect-target": "https://unapproved.example/v1/messages",
            "x-redirect-status": "303",
        },
    )
    with pytest.raises((ValueError, anthropic.APIConnectionError)):
        await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert len(wire_requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [307, 308])
async def test_auxiliary_redirect_validates_destination_before_transport(
    wire_requests: list[Any], status: int, *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="gzip",
        default_headers={
            "x-redirect-target": "https://unapproved.example/v1/messages/count_tokens",
            "x-redirect-status": str(status),
        },
    )
    sync_count = model._client.messages.count_tokens
    async_count = model._async_client.messages.count_tokens
    with pytest.raises((ValueError, anthropic.APIConnectionError)):
        await async_count(
            model="claude-test", messages=[{"role": "user", "content": "hello"}]
        ) if asynchronous else sync_count(
            model="claude-test", messages=[{"role": "user", "content": "hello"}]
        )
    assert len(wire_requests) == 1
    assert wire_requests[0].url.path == "/v1/messages/count_tokens"
    assert "Content-Encoding" not in wire_requests[0].headers
    assert json.loads(wire_requests[0].content)["model"] == "claude-test"
