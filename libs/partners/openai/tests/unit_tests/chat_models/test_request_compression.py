"""Request-body compression contracts through the real OpenAI SDK."""

from __future__ import annotations

import json
import socket
import sys
from collections.abc import AsyncIterator
from typing import Any

import openai
import pytest
import zstandard
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables.configurable import RunnableConfigurableFields
from pydantic import BaseModel, SecretStr, TypeAdapter

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai._compat import httpx
from langchain_openai.chat_models import _client_utils

_RESPONSE = {
    "id": "resp_test",
    "object": "response",
    "created_at": 1234567890,
    "model": "gpt-test",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [
        {
            "type": "message",
            "id": "msg_test",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        }
    ],
}


@pytest.fixture
def transport_options() -> list[dict[str, Any]]:
    return []


@pytest.fixture
async def wire_requests(
    monkeypatch: pytest.MonkeyPatch, transport_options: list[dict[str, Any]]
) -> AsyncIterator[list[Any]]:
    requests: list[Any] = []
    sync_clients: list[Any] = []
    async_clients: list[Any] = []
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    for variable in (
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_PROXY",
        "LANGSMITH_GATEWAY",
        "LANGSMITH_GATEWAY_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)
    _client_utils._cached_sync_httpx_client.cache_clear()
    _client_utils._cached_async_httpx_client.cache_clear()

    def handler(request: Any) -> Any:
        requests.append(request)
        if request.headers.get("x-redirect-target") and len(requests) == 1:
            return httpx.Response(
                int(request.headers["x-redirect-status"]),
                headers={"Location": request.headers["x-redirect-target"]},
            )
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 9}
            )
        if request.headers.get("x-retry-test") and len(requests) == 1:
            return httpx.Response(
                429,
                headers={"retry-after-ms": "1"},
                json={
                    "error": {
                        "message": "retry",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"answer":"hello"}',
                                "refusal": None,
                            },
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                },
            )
        content = (
            zstandard.ZstdDecompressor().decompress(request.content)
            if request.headers.get("Content-Encoding") == "zstd"
            else request.content
        )
        if json.loads(content).get("stream"):
            events = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**_RESPONSE, "status": "in_progress", "output": []},
                },
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "content_index": 0,
                    "output_index": 0,
                    "item_id": "msg_test",
                    "delta": "hello",
                    "logprobs": [],
                },
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": _RESPONSE,
                },
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

    original_proxy_sync = _client_utils._SyncProxyHttpxClientWrapper.__init__
    original_proxy_async = _client_utils._AsyncProxyHttpxClientWrapper.__init__
    original_sync = _client_utils._SyncHttpxClientWrapper.__init__
    original_async = _client_utils._AsyncHttpxClientWrapper.__init__
    original_transport = httpx.HTTPTransport.__init__
    original_async_transport = httpx.AsyncHTTPTransport.__init__

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

    def initialize_proxy_sync(self: Any, **kwargs: Any) -> None:
        original_proxy_sync(self, **kwargs)
        sync_clients.append(self)

    def initialize_proxy_async(self: Any, **kwargs: Any) -> None:
        original_proxy_async(self, **kwargs)
        async_clients.append(self)

    def initialize_transport(self: Any, **kwargs: Any) -> None:
        transport_options.append(kwargs)
        original_transport(self, **kwargs)

    def initialize_async_transport(self: Any, **kwargs: Any) -> None:
        transport_options.append(kwargs)
        original_async_transport(self, **kwargs)

    def send_transport(self: Any, request: Any) -> Any:
        return handler(request)

    async def send_async_transport(self: Any, request: Any) -> Any:
        return handler(request)

    monkeypatch.setattr(
        _client_utils._SyncHttpxClientWrapper, "__init__", initialize_sync
    )
    monkeypatch.setattr(
        _client_utils._AsyncHttpxClientWrapper, "__init__", initialize_async
    )
    monkeypatch.setattr(
        _client_utils._SyncProxyHttpxClientWrapper, "__init__", initialize_proxy_sync
    )
    monkeypatch.setattr(
        _client_utils._AsyncProxyHttpxClientWrapper, "__init__", initialize_proxy_async
    )
    monkeypatch.setattr(httpx.HTTPTransport, "__init__", initialize_transport)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "__init__", initialize_async_transport
    )
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", send_transport)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", send_async_transport
    )
    yield requests
    for client in sync_clients:
        client.close()
    for client in async_clients:
        await client.aclose()
    _client_utils._cached_sync_httpx_client.cache_clear()
    _client_utils._cached_async_httpx_client.cache_clear()


def make_model(**options: Any) -> ChatOpenAI:
    defaults: dict[str, Any] = {
        "model": "gpt-test",
        "api_key": "unit-test-key",
        "use_responses_api": True,
        "max_retries": 0,
        "http_socket_options": (),
    }
    return ChatOpenAI(**{**defaults, **options})


def assert_wire(request: Any, compression: str | None) -> dict[str, Any]:
    assert request.url.host == "api.openai.com"
    assert request.url.path == "/v1/responses"
    if compression:
        assert request.headers["Content-Encoding"] == "zstd"
        content = zstandard.ZstdDecompressor().decompress(request.content)
    else:
        assert "Content-Encoding" not in request.headers
        content = request.content
    assert int(request.headers["Content-Length"]) == len(request.content)
    assert request.headers["Content-Type"].startswith("application/json")
    payload: dict[str, Any] = json.loads(content)
    assert "request_compression" not in payload
    assert payload["model"] == "gpt-test"
    assert payload["input"][0]["role"] == "user"
    return payload


@pytest.mark.parametrize("compression", [None, "zstd"])
def test_invoke_sends_serialized_json_with_selected_encoding(
    wire_requests: list[Any], compression: str | None
) -> None:
    assert (
        make_model(
            **({} if compression is None else {"request_compression": compression})
        )
        .invoke("hello")
        .text
        == "hello"
    )
    assert_wire(wire_requests[0], compression)


@pytest.mark.parametrize("compression", [None, "zstd"])
async def test_ainvoke_sends_serialized_json(
    wire_requests: list[Any], compression: str | None
) -> None:
    assert (
        await make_model(
            **({} if compression is None else {"request_compression": compression})
        ).ainvoke("hello")
    ).text == "hello"
    assert_wire(wire_requests[0], compression)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_preserves_sse_response(
    wire_requests: list[Any], asynchronous: bool
) -> None:
    model = make_model(request_compression="zstd")
    chunks = (
        [chunk async for chunk in model.astream("hello")]
        if asynchronous
        else list(model.stream("hello"))
    )
    assert "".join(str(chunk.text) for chunk in chunks) == "hello"
    assert assert_wire(wire_requests[0], "zstd")["stream"] is True


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_sdk_retry_compresses_once_per_attempt(
    wire_requests: list[Any], asynchronous: bool
) -> None:
    model = make_model(
        request_compression="zstd", max_retries=1, default_headers={"x-retry-test": "1"}
    )
    result = await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert result.text == "hello"
    assert len(wire_requests) == 2
    for request in wire_requests:
        assert_wire(request, "zstd")
    assert wire_requests[0].content == wire_requests[1].content


@pytest.mark.parametrize("codec", ["gzip", "br", True])
def test_rejects_unsupported_codec(wire_requests: list[Any], codec: Any) -> None:
    with pytest.raises(ValueError, match="request_compression"):
        make_model(request_compression=codec)
    assert wire_requests == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.openai.com/v1",
        "https://api.openai.com.evil.test/v1",
        "https://api.openai.com:444/v1",
        "https://user@api.openai.com/v1",
        "https://api.openai.com/custom",
        "https://example.openai.azure.com/openai",
        "https://bedrock-mantle.us-east-1.api.aws/v1",
    ],
)
def test_rejects_unsupported_destination(
    wire_requests: list[Any], base_url: str
) -> None:
    with pytest.raises(ValueError, match=r"compression|api.openai.com"):
        make_model(request_compression="zstd", base_url=base_url).invoke("hello")
    assert wire_requests == []


@pytest.mark.parametrize("responses", [None, False])
def test_rejects_chat_completions_routing(
    wire_requests: list[Any], responses: bool | None
) -> None:
    with pytest.raises(ValueError, match=r"Responses|use_responses_api"):
        make_model(request_compression="zstd", use_responses_api=responses).invoke(
            "hello"
        )
    assert wire_requests == []


def test_rejects_caller_owned_http_client(wire_requests: list[Any]) -> None:
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
            trust_env=False,
        ) as client,
        pytest.raises(ValueError, match="client"),
    ):
        make_model(request_compression="zstd", http_client=client)
    assert wire_requests == []


def test_encoding_conflict_fails_before_send(wire_requests: list[Any]) -> None:
    with pytest.raises(ValueError, match=r"Content-Encoding|content-encoding"):
        make_model(
            request_compression="zstd", default_headers={"content-encoding": "gzip"}
        ).invoke("hello")
    assert wire_requests == []


def test_length_headers_and_extensions_are_preserved(wire_requests: list[Any]) -> None:
    make_model(
        request_compression="zstd",
        timeout=2,
        default_headers={"content-length": "999", "x-custom": "preserved"},
    ).invoke("hello")
    request = wire_requests[0]
    assert_wire(request, "zstd")
    assert request.headers["x-custom"] == "preserved"
    assert request.headers["authorization"] == "Bearer unit-test-key"
    assert request.extensions["timeout"]["read"] == 2


def test_cache_isolates_enabled_and_default_clients(wire_requests: list[Any]) -> None:
    enabled = make_model(request_compression="zstd")
    disabled = make_model()
    enabled.invoke("hello")
    disabled.invoke("hello")
    make_model(request_compression="zstd").invoke("hello")
    assert [request.headers.get("Content-Encoding") for request in wire_requests] == [
        "zstd",
        None,
        "zstd",
    ]
    assert enabled.root_client is not None
    assert disabled.root_client is not None
    assert enabled.root_client._client is not disabled.root_client._client


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("from_environment", [False, True])
@pytest.mark.parametrize("tuned_sockets", [False, True])
async def test_owned_proxy_clients_compress_without_losing_proxy_options(
    wire_requests: list[Any],
    transport_options: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    from_environment: bool,
    tuned_sockets: bool,
) -> None:
    proxy = "http://proxy.example:8080"
    options: dict[str, Any] = {
        "http_socket_options": ((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),)
        if tuned_sockets
        else ()
    }
    if from_environment:
        monkeypatch.setenv("OPENAI_PROXY", proxy)
    else:
        options["openai_proxy"] = proxy
    model = make_model(request_compression="zstd", **options)
    result = await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert result.text == "hello"
    assert_wire(wire_requests[0], "zstd")
    proxied = [
        options
        for options in transport_options
        if "proxy.example" in str(options.get("proxy", ""))
    ]
    assert proxied
    if tuned_sockets:
        assert any(
            options.get("socket_options")
            == [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
            for options in proxied
        )


def test_missing_codec_errors_only_when_enabled(
    wire_requests: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "zstandard", None)
    assert make_model().invoke("hello").text == "hello"
    wire_requests.clear()
    with pytest.raises(ImportError, match="zstandard"):
        make_model(request_compression="zstd").invoke("hello")
    assert wire_requests == []


@pytest.mark.parametrize("delta", [-1, 0])
def test_decompressed_size_limit_boundary(
    wire_requests: list[Any], monkeypatch: pytest.MonkeyPatch, delta: int
) -> None:
    make_model().invoke("hello")
    raw_size = len(wire_requests[0].content)
    wire_requests.clear()
    monkeypatch.setattr(_client_utils, "_MAX_REQUEST_BYTES", raw_size + delta)
    if delta < 0:
        with pytest.raises(ValueError, match="128 MiB"):
            make_model(request_compression="zstd").invoke("hello")
        assert wire_requests == []
    else:
        assert make_model(request_compression="zstd").invoke("hello").text == "hello"
        assert_wire(wire_requests[0], "zstd")


def test_compressed_size_limit_is_checked(
    wire_requests: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"model": "gpt-test", "input": "hi"}
    raw = httpx.Request(
        "POST", "https://api.openai.com/v1/responses", json=payload
    ).content
    assert len(zstandard.ZstdCompressor(level=3).compress(raw)) > len(raw)
    monkeypatch.setattr(_client_utils, "_MAX_REQUEST_BYTES", len(raw))
    client = _client_utils._get_default_httpx_client(
        "https://api.openai.com/v1", None, (), "zstd"
    )
    with pytest.raises(ValueError, match=r"compressed.*128 MiB"):
        client.build_request(
            "POST", "https://api.openai.com/v1/responses", json=payload
        )
    assert wire_requests == []


def test_ratio_limit_has_no_uncompressed_fallback(wire_requests: list[Any]) -> None:
    with pytest.raises(ValueError, match=r"100x|ratio"):
        make_model(request_compression="zstd").invoke("a" * 100000)
    assert wire_requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("tuned_sockets", [False, True])
async def test_proxy_client_defaults_match_without_compression(
    wire_requests: list[Any], asynchronous: bool, tuned_sockets: bool
) -> None:
    sockets = ((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),) if tuned_sockets else ()
    builder = (
        _client_utils._build_proxied_async_httpx_client
        if asynchronous
        else _client_utils._build_proxied_sync_httpx_client
    )
    disabled: Any = builder("http://proxy.example:8080", True, sockets)
    enabled: Any = builder("http://proxy.example:8080", True, sockets, "zstd")
    try:

        def settings(client: Any) -> tuple[Any, ...]:
            transport = next(
                value for value in client._mounts.values() if value is not None
            )
            pool = transport._pool
            return (
                client.timeout,
                client.follow_redirects,
                pool._max_connections,
                pool._max_keepalive_connections,
            )

        assert settings(enabled) == settings(disabled)
    finally:
        if asynchronous:
            await disabled.aclose()
            await enabled.aclose()
        else:
            disabled.close()
            enabled.close()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("OPENAI_API_BASE", "https://proxy.example/v1"),
        ("OPENAI_BASE_URL", "https://proxy.example/v1"),
        ("LANGSMITH_GATEWAY", "true"),
    ],
)
def test_environment_destination_is_validated(
    wire_requests: list[Any], monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="compression"):
        make_model(request_compression="zstd").invoke("hello")
    assert wire_requests == []


@pytest.mark.parametrize(
    "field",
    ["client", "async_client", "root_client", "root_async_client", "http_async_client"],
)
def test_caller_owned_clients_are_rejected(
    wire_requests: list[Any], field: str
) -> None:
    with pytest.raises(ValueError, match="client"):
        make_model(request_compression="zstd", **{field: object()})
    assert wire_requests == []


def test_azure_compression_is_rejected(wire_requests: list[Any]) -> None:
    with pytest.raises(ValueError, match=r"Azure|compression"):
        AzureChatOpenAI(
            model="gpt-test",
            api_key=SecretStr("unit-test-key"),
            azure_endpoint="https://resource.openai.azure.com",
            api_version="2025-01-01-preview",
            use_responses_api=True,
            request_compression="zstd",
        )
    assert wire_requests == []


def test_wrapped_sdk_chat_completions_is_rejected(wire_requests: list[Any]) -> None:
    model = make_model(request_compression="zstd")
    assert model.root_client is not None
    with pytest.raises(ValueError, match="Responses"):
        model.root_client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hello"}]
        )
    assert wire_requests == []


def test_auxiliary_token_count_request_stays_uncompressed(
    wire_requests: list[Any],
) -> None:
    model = make_model(request_compression="zstd")
    assert model.root_client is not None
    result = model.root_client.responses.input_tokens.count(
        model="gpt-test", input="hello"
    )
    assert result.input_tokens == 9
    assert "Content-Encoding" not in wire_requests[0].headers
    assert json.loads(wire_requests[0].content)["input"] == "hello"


def test_non_json_inference_request_is_rejected(wire_requests: list[Any]) -> None:
    model = make_model(request_compression="zstd")
    assert model.root_client is not None
    with pytest.raises(ValueError, match="JSON"):
        model.root_client._client.build_request(
            "POST",
            "https://api.openai.com/v1/responses",
            content=b"opaque",
            headers={"Content-Type": "application/octet-stream"},
        )
    assert wire_requests == []


def test_unbuffered_json_request_fails_clearly(wire_requests: list[Any]) -> None:
    model = make_model(request_compression="zstd")
    assert model.root_client is not None
    with pytest.raises(ValueError, match=r"buffered|JSON"):
        model.root_client._client.build_request(
            "POST",
            "https://api.openai.com/v1/responses",
            content=iter([b'{"model":"test"}']),
            headers={"Content-Type": "application/json"},
        )
    assert wire_requests == []


@pytest.mark.parametrize("compression", [None, "zstd"])
def test_configurable_model_reconstruction_preserves_owned_client_policy(
    wire_requests: list[Any], compression: str | None
) -> None:
    model = make_model(request_compression=compression).configurable_fields(
        temperature=ConfigurableField(id="temperature")
    )
    result = model.invoke("hello", {"configurable": {"temperature": 0.5}})
    assert result.text == "hello"
    payload = assert_wire(wire_requests[0], compression)
    assert payload["temperature"] == 0.5


@pytest.mark.parametrize(("initial", "configured"), [(None, "zstd"), ("zstd", None)])
def test_configurable_codec_changes_rebuild_owned_clients(
    wire_requests: list[Any], initial: str | None, configured: str | None
) -> None:
    original = make_model(request_compression=initial)
    original_root = original.root_client
    model = original.configurable_fields(
        request_compression=ConfigurableField(id="compression")
    )
    assert (
        model.invoke("hello", {"configurable": {"compression": configured}}).text
        == "hello"
    )
    assert_wire(wire_requests[0], configured)
    assert original.root_client is original_root
    assert original.invoke("hello").text == "hello"
    assert_wire(wire_requests[1], initial)


def test_configurable_timeout_rebuilds_owned_clients(wire_requests: list[Any]) -> None:
    model = make_model(request_compression="zstd", timeout=2).configurable_fields(
        request_timeout=ConfigurableField(id="timeout")
    )
    assert model.invoke("hello", {"configurable": {"timeout": 7}}).text == "hello"
    assert_wire(wire_requests[0], "zstd")
    assert wire_requests[0].extensions["timeout"]["read"] == 7


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicitly_reused_sdk_clients_preserve_endpoint_and_credentials(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    original = make_model(
        api_key="fake-original", base_url="https://original.example/v1"
    )
    reused = make_model(
        api_key="fake-new",
        client=original.client,
        async_client=original.async_client,
        root_client=original.root_client,
        root_async_client=original.root_async_client,
    )
    result = await reused.ainvoke("hello") if asynchronous else reused.invoke("hello")
    assert result.text == "hello"
    assert wire_requests[0].url.host == "original.example"
    assert wire_requests[0].headers["Authorization"] == "Bearer fake-original"
    assert reused.client is original.client
    assert reused.async_client is original.async_client
    assert reused.root_client is original.root_client
    assert reused.root_async_client is original.root_async_client


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicitly_reused_proxy_http_clients_preserve_identity(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    original = make_model(openai_proxy="http://proxy.example:8080")
    reused = make_model(
        http_client=original.http_client,
        http_async_client=original.http_async_client,
    )
    result = await reused.ainvoke("hello") if asynchronous else reused.invoke("hello")
    assert result.text == "hello"
    assert original.http_client is not None
    assert original.http_async_client is not None
    original.http_client.close()
    await original.http_async_client.aclose()
    assert reused.http_client is original.http_client
    assert reused.http_async_client is original.http_async_client
    assert reused.http_client is not None
    assert reused.http_async_client is not None
    assert_wire(wire_requests[0], None)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize(
    "target",
    [
        "https://unapproved.example/v1/responses",
        "https://api.openai.com/v1/chat/completions",
    ],
)
async def test_redirect_cannot_send_compressed_body_to_unapproved_destination(
    wire_requests: list[Any], status: int, target: str, *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="zstd",
        default_headers={"x-redirect-target": target, "x-redirect-status": str(status)},
    )
    with pytest.raises((ValueError, openai.APIConnectionError)):
        await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert len(wire_requests) == 1
    assert_wire(wire_requests[0], "zstd")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [307, 308])
async def test_supported_query_redirect_preserves_single_compressed_body(
    wire_requests: list[Any], status: int, *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="zstd",
        default_headers={
            "x-redirect-target": "https://api.openai.com/v1/responses?redirected=true",
            "x-redirect-status": str(status),
        },
    )
    result = await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert result.text == "hello"
    assert len(wire_requests) == 2
    assert_wire(wire_requests[0], "zstd")
    assert_wire(wire_requests[1], "zstd")
    assert wire_requests[0].content == wire_requests[1].content
    assert wire_requests[1].url.params["redirected"] == "true"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_method_rewriting_redirect_is_rejected(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="zstd",
        default_headers={
            "x-redirect-target": "https://unapproved.example/v1/responses",
            "x-redirect-status": "303",
        },
    )
    with pytest.raises((ValueError, openai.APIConnectionError)):
        await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    assert len(wire_requests) == 1


@pytest.mark.parametrize(
    "field",
    [
        "client",
        "async_client",
        "root_client",
        "root_async_client",
        "http_client",
        "http_async_client",
    ],
)
async def test_explicit_configurable_original_client_override_is_not_discarded(
    wire_requests: list[Any], field: str
) -> None:
    original = make_model(openai_proxy="http://proxy.example:8080")
    supplied = getattr(original, field)
    assert supplied is not None
    configurable = original.configurable_fields(
        request_compression=ConfigurableField(id="compression"),
        **{field: ConfigurableField(id="supplied")},
    )
    try:
        with pytest.raises(ValueError, match="client"):
            configurable.invoke(
                "hello", {"configurable": {"compression": "zstd", "supplied": supplied}}
            )
        assert wire_requests == []
    finally:
        assert original.http_client is not None
        original.http_client.close()
        assert original.http_async_client is not None
        await original.http_async_client.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_mixed_caller_http_client_survives_disabled_reconstruction(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    sync_client = httpx.Client(trust_env=False)
    async_client = httpx.AsyncClient(trust_env=False)
    try:
        original = make_model(http_client=sync_client, http_async_client=async_client)
        configurable = original.configurable_fields(
            temperature=ConfigurableField(id="temperature")
        )
        assert isinstance(configurable, RunnableConfigurableFields)
        prepared, config = configurable.prepare({"configurable": {"temperature": 0.5}})
        assert isinstance(prepared, ChatOpenAI)
        assert prepared.http_client is sync_client
        assert prepared.http_async_client is async_client
        result = (
            await prepared.ainvoke("hello", config)
            if asynchronous
            else prepared.invoke("hello", config)
        )
        assert result.text == "hello"
        assert_wire(wire_requests[0], None)
        assert original.http_client is sync_client
        assert original.http_async_client is async_client
        with pytest.raises(ValueError, match="client"):
            original.configurable_fields(
                request_compression=ConfigurableField(id="compression")
            ).invoke("hello", {"configurable": {"compression": "zstd"}})
        assert len(wire_requests) == 1
    finally:
        sync_client.close()
        await async_client.aclose()


@pytest.mark.parametrize("field", ["http_client", "http_async_client"])
async def test_replaced_client_is_not_discarded_by_stale_generated_provenance(
    wire_requests: list[Any], field: str
) -> None:
    original = make_model(openai_proxy="http://proxy.example:8080")
    old_sync = original.http_client
    old_async = original.http_async_client
    replacement = (
        httpx.Client(trust_env=False)
        if field == "http_client"
        else httpx.AsyncClient(trust_env=False)
    )
    setattr(original, field, replacement)
    try:
        configurable = original.configurable_fields(
            request_compression=ConfigurableField(id="compression")
        )
        with pytest.raises(ValueError, match="client"):
            configurable.invoke("hello", {"configurable": {"compression": "zstd"}})
        assert getattr(original, field) is replacement
        assert wire_requests == []
    finally:
        if isinstance(replacement, httpx.AsyncClient):
            await replacement.aclose()
        else:
            replacement.close()
        assert old_sync is not None
        assert old_async is not None
        old_sync.close()
        await old_async.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [307, 308])
async def test_auxiliary_redirect_validates_destination_before_transport(
    wire_requests: list[Any], status: int, *, asynchronous: bool
) -> None:
    model = make_model(
        request_compression="zstd",
        default_headers={
            "x-redirect-target": "https://unapproved.example/v1/responses/input_tokens",
            "x-redirect-status": str(status),
        },
    )
    assert model.root_client is not None
    assert model.root_async_client is not None
    sync_count = model.root_client.responses.input_tokens.count
    async_count = model.root_async_client.responses.input_tokens.count
    with pytest.raises((ValueError, openai.APIConnectionError)):
        await async_count(
            model="gpt-test", input="hello"
        ) if asynchronous else sync_count(model="gpt-test", input="hello")
    assert len(wire_requests) == 1
    assert wire_requests[0].url.path == "/v1/responses/input_tokens"
    assert "Content-Encoding" not in wire_requests[0].headers
    assert json.loads(wire_requests[0].content)["model"] == "gpt-test"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_configurable_explicit_root_override_preserves_response_routing(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    original = make_model(api_key="fake-new")
    supplied_model = make_model(
        api_key="fake-original", base_url="https://original.example/v1"
    )
    field = "root_async_client" if asynchronous else "root_client"
    supplied_root = getattr(supplied_model, field)
    assert supplied_root is not None
    configurable = original.configurable_fields(**{field: ConfigurableField(id="root")})
    assert isinstance(configurable, RunnableConfigurableFields)
    prepared, config = configurable.prepare({"configurable": {"root": supplied_root}})
    assert isinstance(prepared, ChatOpenAI)
    result = (
        await prepared.ainvoke("hello", config)
        if asynchronous
        else prepared.invoke("hello", config)
    )
    assert result.text == "hello"
    assert getattr(prepared, field) is supplied_root
    assert wire_requests[0].url.host == "original.example"
    assert wire_requests[0].headers["Authorization"] == "Bearer fake-original"
    assert "Content-Encoding" not in wire_requests[0].headers


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_configurable_resource_override_preserves_response_root_settings(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    original = make_model(
        api_key="fake-original", base_url="https://original.example/v1"
    )
    original_root = original.root_client
    original_async_root = original.root_async_client
    supplied_model = make_model(
        api_key="fake-other", base_url="https://other.example/v1"
    )
    field = "async_client" if asynchronous else "client"
    supplied_resource = getattr(supplied_model, field)
    assert supplied_resource is not None
    configurable = original.configurable_fields(
        **{field: ConfigurableField(id="resource")}
    )
    assert isinstance(configurable, RunnableConfigurableFields)
    prepared, config = configurable.prepare(
        {"configurable": {"resource": supplied_resource}}
    )
    assert isinstance(prepared, ChatOpenAI)
    result = (
        await prepared.ainvoke("hello", config)
        if asynchronous
        else prepared.invoke("hello", config)
    )
    assert result.text == "hello"
    assert getattr(prepared, field) is supplied_resource
    assert prepared.root_client is not None
    assert prepared.root_async_client is not None
    assert wire_requests[0].url.host == "original.example"
    assert wire_requests[0].headers["Authorization"] == "Bearer fake-original"
    assert "Content-Encoding" not in wire_requests[0].headers
    assert original.root_client is original_root
    assert original.root_async_client is original_async_root
    original_result = (
        await original.ainvoke("hello") if asynchronous else original.invoke("hello")
    )
    assert original_result.text == "hello"
    assert wire_requests[1].url.host == "original.example"
    assert wire_requests[1].headers["Authorization"] == "Bearer fake-original"


def test_chat_completions_resource_only_does_not_build_unused_sdk_roots(
    wire_requests: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sync_resource = object()
    async_resource = object()
    model = make_model(
        api_key=None,
        use_responses_api=False,
        client=sync_resource,
        async_client=async_resource,
    )
    assert model.client is sync_resource
    assert model.async_client is async_resource
    assert model.root_client is None
    assert model.root_async_client is None
    assert wire_requests == []


class CompressionModelHolder(BaseModel):
    model: ChatOpenAI


@pytest.mark.parametrize("compression", [None, "zstd"])
@pytest.mark.parametrize("entry", ["model_validate", "holder", "type_adapter"])
def test_existing_model_revalidation_preserves_identity_and_wire_codec(
    wire_requests: list[Any], compression: str | None, entry: str
) -> None:
    original = make_model(request_compression=compression)
    original_root = original.root_client
    original_async_root = original.root_async_client
    if entry == "model_validate":
        validated = ChatOpenAI.model_validate(original)
    elif entry == "holder":
        validated = CompressionModelHolder.model_validate({"model": original}).model
    else:
        validated = TypeAdapter(ChatOpenAI).validate_python(original)
    assert validated is original
    assert validated.root_client is original_root
    assert validated.root_async_client is original_async_root
    assert validated.invoke("hello").text == "hello"
    assert_wire(wire_requests[0], compression)


@pytest.mark.parametrize("field", ["client", "async_client"])
def test_existing_enabled_model_revalidation_rejects_replaced_clients(
    wire_requests: list[Any], field: str
) -> None:
    original = make_model(request_compression="zstd")
    replacement = object()
    setattr(original, field, replacement)
    with pytest.raises(ValueError, match="client"):
        ChatOpenAI.model_validate(original)
    assert getattr(original, field) is replacement
    assert wire_requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reconstructed", [False, True])
async def test_resource_override_supports_deferred_responses_routing(
    wire_requests: list[Any], *, asynchronous: bool, reconstructed: bool
) -> None:
    original = make_model(
        api_key="fake-original",
        base_url="https://original.example/v1",
        use_responses_api=None,
    )
    original_root = original.root_client
    original_async_root = original.root_async_client
    field = "async_client" if asynchronous else "client"
    expected_resource = getattr(original, field)
    selected = original
    config = None
    if reconstructed:
        supplied_model = make_model(
            api_key="fake-other", base_url="https://other.example/v1"
        )
        expected_resource = getattr(supplied_model, field)
        configurable = original.configurable_fields(
            **{field: ConfigurableField(id="resource")}
        )
        assert isinstance(configurable, RunnableConfigurableFields)
        prepared, config = configurable.prepare(
            {"configurable": {"resource": expected_resource}}
        )
        assert isinstance(prepared, ChatOpenAI)
        selected = prepared
    assert isinstance(selected, ChatOpenAI)
    result = (
        await selected.ainvoke("hello", config, previous_response_id="resp_previous")
        if asynchronous
        else selected.invoke("hello", config, previous_response_id="resp_previous")
    )
    assert result.text == "hello"
    assert getattr(selected, field) is expected_resource
    assert selected.root_client is not None
    assert selected.root_async_client is not None
    assert wire_requests[0].url.host == "original.example"
    assert wire_requests[0].url.path == "/v1/responses"
    assert wire_requests[0].headers["Authorization"] == "Bearer fake-original"
    assert "Content-Encoding" not in wire_requests[0].headers
    assert (
        json.loads(wire_requests[0].content)["previous_response_id"] == "resp_previous"
    )
    assert original.root_client is original_root
    assert original.root_async_client is original_async_root


class StructuredAnswer(BaseModel):
    answer: str


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_resource_override_supports_chat_completions_structured_parse(
    wire_requests: list[Any], *, asynchronous: bool
) -> None:
    original = make_model(
        api_key="fake-original",
        base_url="https://original.example/v1",
        use_responses_api=False,
    )
    supplied_model = make_model(
        api_key="fake-other", base_url="https://other.example/v1"
    )
    field = "async_client" if asynchronous else "client"
    supplied_resource = getattr(supplied_model, field)
    configurable = original.configurable_fields(
        **{field: ConfigurableField(id="resource")}
    )
    assert isinstance(configurable, RunnableConfigurableFields)
    prepared, config = configurable.prepare(
        {"configurable": {"resource": supplied_resource}}
    )
    assert isinstance(prepared, ChatOpenAI)
    structured = prepared.with_structured_output(StructuredAnswer, method="json_schema")
    result = (
        await structured.ainvoke("hello", config)
        if asynchronous
        else structured.invoke("hello", config)
    )
    assert result == StructuredAnswer(answer="hello")
    assert getattr(prepared, field) is supplied_resource
    assert prepared.root_client is not None
    assert prepared.root_async_client is not None
    assert wire_requests[0].url.host == "original.example"
    assert wire_requests[0].url.path == "/v1/chat/completions"
    assert wire_requests[0].headers["Authorization"] == "Bearer fake-original"
    assert "Content-Encoding" not in wire_requests[0].headers
    assert (
        json.loads(wire_requests[0].content)["response_format"]["type"] == "json_schema"
    )
