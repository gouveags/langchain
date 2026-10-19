"""Helpers for creating Anthropic API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatAnthropic.

Logic is largely replicated from anthropic._base_client.
"""

from __future__ import annotations

import asyncio
import gzip
import importlib
import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

import anthropic

_NOT_GIVEN: Any = object()


def _validate_compression_destination(base_url: str) -> None:
    url = urlsplit(base_url)
    if (
        url.scheme != "https"
        or url.hostname != "api.anthropic.com"
        or url.port not in (None, 443)
        or url.username is not None
        or url.password is not None
        or url.path.rstrip("/")
        or url.query
        or url.fragment
    ):
        msg = 'request_compression="gzip" requires https://api.anthropic.com.'
        raise ValueError(msg)


def _validate_compressed_request(request: Any) -> None:
    _validate_compression_destination(str(request.url.copy_with(path="/", query=None)))
    if request.extensions.get("langchain.request_compression") is None:
        return
    if request.method != "POST" or request.url.path != "/v1/messages":
        msg = "request_compression requires POST /v1/messages."
        raise ValueError(msg)
    if request.headers.get("content-encoding") != "gzip":
        msg = "request_compression encoding changed before sending."
        raise ValueError(msg)


async def _validate_compressed_async_request(request: Any) -> None:
    _validate_compressed_request(request)


def _add_compression_hook(kwargs: dict[str, Any], *, asynchronous: bool) -> None:
    hooks = {
        name: list(values) for name, values in (kwargs.get("event_hooks") or {}).items()
    }
    hooks.setdefault("request", []).append(
        _validate_compressed_async_request
        if asynchronous
        else _validate_compressed_request
    )
    kwargs["event_hooks"] = hooks


def _compress_request(request: Any, encoding: str | None) -> Any:
    if encoding is None:
        return request
    if encoding != "gzip":
        msg = 'Anthropic request_compression must be None or "gzip".'
        raise ValueError(msg)
    _validate_compression_destination(str(request.url.copy_with(path="/", query=None)))
    if request.method != "POST" or request.url.path != "/v1/messages":
        return request
    headers = request.headers.copy()
    if "content-encoding" in headers:
        msg = "request_compression cannot be combined with Content-Encoding."
        raise ValueError(msg)
    if headers.get("content-type", "").split(";")[0].strip() != "application/json":
        msg = "request_compression requires a buffered JSON request body."
        raise ValueError(msg)
    httpx = importlib.import_module(type(request).__module__.split(".", 1)[0])
    try:
        body = request.content
    except httpx.RequestNotRead as exc:
        msg = "request_compression requires a buffered JSON request body."
        raise ValueError(msg) from exc
    content = gzip.compress(body, mtime=0)
    headers.pop("content-length", None)
    headers["Content-Encoding"] = encoding
    extensions = request.extensions.copy()
    extensions["langchain.request_compression"] = encoding
    return type(request)(
        method=request.method,
        url=request.url,
        headers=headers,
        content=content,
        extensions=extensions,
    )


class _SyncHttpxClientWrapper(anthropic.DefaultHttpxClient):
    """Borrowed from anthropic._base_client."""

    def __init__(
        self, *, request_compression: str | None = None, **kwargs: Any
    ) -> None:
        if request_compression is not None:
            _add_compression_hook(kwargs, asynchronous=False)
        super().__init__(**kwargs)
        self._request_compression = request_compression

    def build_request(self, method: str, url: Any, **kwargs: Any) -> Any:
        request = super().build_request(method, url, **kwargs)
        return _compress_request(request, self._request_compression)

    def __del__(self) -> None:
        try:
            if self.is_closed:
                return
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(anthropic.DefaultAsyncHttpxClient):
    """Borrowed from anthropic._base_client."""

    def __init__(
        self, *, request_compression: str | None = None, **kwargs: Any
    ) -> None:
        if request_compression is not None:
            _add_compression_hook(kwargs, asynchronous=True)
        super().__init__(**kwargs)
        self._request_compression = request_compression

    def build_request(self, method: str, url: Any, **kwargs: Any) -> Any:
        request = super().build_request(method, url, **kwargs)
        return _compress_request(request, self._request_compression)

    def __del__(self) -> None:
        try:
            if self.is_closed:
                return
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


@lru_cache
def _get_default_httpx_client(
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
    request_compression: str | None = None,
) -> _SyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _SyncHttpxClientWrapper(request_compression=request_compression, **kwargs)


@lru_cache
def _get_default_async_httpx_client(
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
    request_compression: str | None = None,
) -> _AsyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _AsyncHttpxClientWrapper(request_compression=request_compression, **kwargs)
