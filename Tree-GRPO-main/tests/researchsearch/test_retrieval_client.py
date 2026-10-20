# Copyright 2026 ResearchSearch-RL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""检索客户端测试（门禁 T03 客户端侧 / 方案 §4.3 阻塞 6）。

锁定 ``post_retrieve`` 的超时、有限重试与错误分类语义：

- 单次请求 ``timeout=30``，可重试错误最多重试 ``max_retries=2`` 次（共 3 次调用）。
- 鉴权失败（401/403）与非法返回（非 2xx 其他、非法 JSON、缺 result）**立即失败不重试**。
- 真实零结果（HTTP 200 + ``result=[]``）不是错误。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load_retrieval_client():
    spec = importlib.util.spec_from_file_location(
        "researchsearch_retrieval_client", _REPO / "search_r1/search/retrieval_client.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_retrieval_client()
RetrievalError = rc.RetrievalError
post_retrieve = rc.post_retrieve


class _FakeResponse:
    def __init__(self, status_code=200, body=None, json_error=None):
        self.status_code = status_code
        self._body = body
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body


_URL = "http://retriever/retrieve"
_PAYLOAD = {"queries": ["q"], "topk": 3, "return_scores": True}


def _post_side_effect(responses):
    return lambda *args, **kwargs: responses.pop(0)


# --- 成功与真实零结果 -------------------------------------------------------


def test_success_returns_body():
    with patch("requests.post", return_value=_FakeResponse(200, {"result": [["doc"]]})):
        body = post_retrieve(_URL, _PAYLOAD)
    assert body == {"result": [["doc"]]}


def test_empty_result_is_not_an_error():
    # 真实零结果（HTTP 200 + result=[]）正常返回，不是错误。
    with patch("requests.post", return_value=_FakeResponse(200, {"result": []})):
        body = post_retrieve(_URL, _PAYLOAD)
    assert body == {"result": []}


# --- 网络失败与可重试错误 ---------------------------------------------------


@pytest.mark.parametrize(
    "exc,kind",
    [
        (__import__("requests").exceptions.Timeout, "network"),
        (__import__("requests").exceptions.ConnectionError, "network"),
    ],
)
def test_network_errors_retry_then_raise(exc, kind):
    with patch("requests.post", side_effect=exc()) as mock_post, patch("time.sleep"):
        with pytest.raises(RetrievalError) as info:
            post_retrieve(_URL, _PAYLOAD)
    assert info.value.kind == kind
    # 1 次初始 + 2 次重试 = 3 次调用。
    assert mock_post.call_count == 3


@pytest.mark.parametrize(
    "status,kind",
    [(429, "rate_limit"), (500, "server"), (503, "server")],
)
def test_retryable_status_retries_then_raise(status, kind):
    with patch("requests.post", return_value=_FakeResponse(status)) as mock_post, patch("time.sleep"):
        with pytest.raises(RetrievalError) as info:
            post_retrieve(_URL, _PAYLOAD)
    assert info.value.kind == kind
    assert mock_post.call_count == 3


def test_retry_then_success():
    responses = [_FakeResponse(503), _FakeResponse(200, {"result": [["ok"]]})]
    with patch("requests.post", side_effect=_post_side_effect(responses)) as mock_post, patch("time.sleep"):
        body = post_retrieve(_URL, _PAYLOAD)
    assert body == {"result": [["ok"]]}
    assert mock_post.call_count == 2


def test_backoff_is_exponential():
    sleeps = []
    with patch("requests.post", return_value=_FakeResponse(429)), patch("time.sleep", side_effect=sleeps.append):
        with pytest.raises(RetrievalError):
            post_retrieve(_URL, _PAYLOAD)
    # 默认 backoff=1.0，两次重试间隔 1.0、2.0（指数退避）。
    assert sleeps == [1.0, 2.0]


# --- 立即失败（不重试）的错误 -------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_auth_fails_immediately(status):
    with patch("requests.post", return_value=_FakeResponse(status)) as mock_post:
        with pytest.raises(RetrievalError) as info:
            post_retrieve(_URL, _PAYLOAD)
    assert info.value.kind == "auth"
    assert mock_post.call_count == 1  # 鉴权错误不重试


def test_other_4xx_is_invalid_and_no_retry():
    with patch("requests.post", return_value=_FakeResponse(400)) as mock_post:
        with pytest.raises(RetrievalError) as info:
            post_retrieve(_URL, _PAYLOAD)
    assert info.value.kind == "invalid"
    assert mock_post.call_count == 1


def test_invalid_json_is_invalid_and_no_retry():
    import requests

    with patch(
        "requests.post",
        return_value=_FakeResponse(200, body=None, json_error=requests.exceptions.JSONDecodeError("bad", "doc", 0)),
    ) as mock_post:
        with pytest.raises(RetrievalError) as info:
            post_retrieve(_URL, _PAYLOAD)
    assert info.value.kind == "invalid"
    assert mock_post.call_count == 1


@pytest.mark.parametrize("body", [["not", "a", "dict"], {"no_result_key": 1}])
def test_malformed_body_is_invalid_and_no_retry(body):
    with patch("requests.post", return_value=_FakeResponse(200, body)) as mock_post:
        with pytest.raises(RetrievalError) as info:
            post_retrieve(_URL, _PAYLOAD)
    assert info.value.kind == "invalid"
    assert mock_post.call_count == 1


# --- 超时与重试上限参数化 ---------------------------------------------------


def test_max_retries_controls_total_attempts():
    with patch("requests.post", return_value=_FakeResponse(429)) as mock_post, patch("time.sleep"):
        with pytest.raises(RetrievalError):
            post_retrieve(_URL, _PAYLOAD, max_retries=0)
    assert mock_post.call_count == 1
