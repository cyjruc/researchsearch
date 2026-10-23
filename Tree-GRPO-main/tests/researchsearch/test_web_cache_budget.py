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
"""Web 检索服务缓存命名空间与预算口径测试（方案 Day 6 任务 3 / 门禁 T10 无卡部分）。

锁定 ``bing_search_server_redis.py`` 的 ``OnlineSearchEngine``：

- **缓存命名空间（B-5）**：缓存键由 provider/zone/版本/max_results/result_length/query
  组成，换任一配置后不再命中旧缓存。
- **预算口径（B-6）**：``logical_queries``（Agent 逻辑搜索，每个 query 一次）与
  ``cache_hits``（Redis 命中）与 ``api_requests``（真实 HTTP 请求，含重试）三者分开统计；
  ``execute_with_retry`` 达到 ``tool_retry_count`` 上限即终止，不无限重试。
- **结构化失败状态（缺口 2）**：``execute`` 返回文本保持兼容，``get_last_outcome()``
  暴露 ``status``（success/not_found/rate_limit/network/server/invalid/auth/empty_query）
  与 ``error_code``，鉴权错误立即失败不重试。
- **URL/标题/rank 旁路（缺口 3）**：``SearchOutcome.items`` 保留每条 organic 的
  rank/title/url/description，模型可见文本仍只含摘要。

本机缺 ``redis``/``fastapi``/``langid``/``uvicorn``，本文件 mock 掉这四个模块，
用 dict 后端 FakeRedis 与 ``mock`` 掉的 ``requests.post`` 做 fixture，无需 GPU 或真实服务。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


class _RedisConnectionError(Exception):
    pass


def _load_bing_server():
    """mock 掉重依赖后真实加载 bing_search_server_redis.py。"""
    mocks = (
        (
            "redis",
            {
                "Redis": type("Redis", (), {}),
                "exceptions": SimpleNamespace(
                    ConnectionError=_RedisConnectionError,
                    BusyLoadingError=_RedisConnectionError,
                ),
            },
        ),
        ("fastapi", {"FastAPI": lambda *a, **k: SimpleNamespace()}),
        ("langid", {"classify": lambda q: ("en", 1.0)}),
        ("uvicorn", {"run": lambda *a, **k: None}),
    )
    for name, attrs in mocks:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for key, value in attrs.items():
            setattr(mod, key, value)

    spec = importlib.util.spec_from_file_location(
        "search_r1.search.bing_search_server_redis",
        _REPO / "search_r1/search/bing_search_server_redis.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["search_r1.search.bing_search_server_redis"] = mod
    spec.loader.exec_module(mod)
    return mod


_server = _load_bing_server()
OnlineSearchEngine = _server.OnlineSearchEngine


class _FakeRedis:
    """dict 后端假 Redis，仅实现 get/set，记录 set 的键。"""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value


class _FakeResponse:
    def __init__(self, json_body=None, status_code=200):
        self.status_code = status_code
        self._json_body = json_body

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = SimpleNamespace(status_code=self.status_code, text="")
            raise err


_SUCCESS_BODY = {"organic": [{"description": "snippet A"}, {"description": "snippet B"}]}


def _make_engine(**kwargs):
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("redis_client", _FakeRedis())
    return OnlineSearchEngine(**kwargs)


# --- B-5 缓存命名空间 -------------------------------------------------------


def test_cache_key_includes_full_namespace():
    engine = _make_engine()
    assert engine._cache_key("who") == "brightdata:serp_api1:v1:10:1000:who"


def test_cache_key_differs_by_provider_zone_version_topk():
    base = _make_engine()
    variants = [
        _make_engine(provider="other"),
        _make_engine(zone="serp_api2"),
        _make_engine(cache_version="v2"),
        _make_engine(max_results=5),
        _make_engine(result_length=500),
    ]
    keys = {base._cache_key("q")} | {v._cache_key("q") for v in variants}
    assert len(keys) == 6  # 六种配置的键互不相同


def test_cache_key_stable_for_same_config():
    a = _make_engine()
    b = _make_engine()
    assert a._cache_key("q") == b._cache_key("q")


def test_namespaced_cache_isolates_providers():
    # 同一 Redis 实例上，provider=A 写入的缓存不能被 provider=B 命中。
    shared = _FakeRedis()
    a = _make_engine(redis_client=shared, provider="A")
    b = _make_engine(redis_client=shared, provider="B")

    with patch("requests.post", return_value=_FakeResponse(_SUCCESS_BODY)) as post_a:
        a.execute("q")
    assert post_a.call_count == 1

    # A 第二次查询命中自己的缓存，不再调 API。
    with patch("requests.post") as post_a2:
        res = a.execute("q")
    assert post_a2.call_count == 0
    assert "snippet" in res

    # B 查同 query：namespace 不同 → miss → 走 API。
    with patch("requests.post", return_value=_FakeResponse(_SUCCESS_BODY)) as post_b:
        b.execute("q")
    assert post_b.call_count == 1


# --- B-6 预算口径统计 -------------------------------------------------------


def test_cache_hit_counts_cache_hits_not_api():
    engine = _make_engine()
    # 用正确的命名空间键预填充缓存。
    engine.redis_client.set(engine._cache_key("q"), "cached result")

    with patch("requests.post") as post:
        res = engine.execute("q")

    assert res == "cached result"
    assert post.call_count == 0
    assert engine.get_stats() == {"logical_queries": 0, "cache_hits": 1, "api_requests": 0}


def test_cache_miss_counts_api_request():
    engine = _make_engine()

    with patch("requests.post", return_value=_FakeResponse(_SUCCESS_BODY)):
        res = engine.execute("q")

    assert res.startswith("Page 1")
    assert engine.get_stats() == {"logical_queries": 0, "cache_hits": 0, "api_requests": 1}


def test_retry_stops_at_limit_and_counts_api_requests():
    engine = _make_engine(tool_retry_count=3)

    with patch("requests.post", side_effect=requests.exceptions.Timeout()) as post, patch("time.sleep"):
        res = engine.execute_with_retry("q")

    # 达到上限终止，不无限重试。
    assert res == "Search failed for query: q"
    assert post.call_count == 3
    # 一个逻辑搜索；三次真实 HTTP 请求（含重试）；零缓存命中。
    assert engine.get_stats() == {"logical_queries": 1, "cache_hits": 0, "api_requests": 3}


def test_retry_then_success_counts_only_real_requests():
    engine = _make_engine(tool_retry_count=3)
    responses = [requests.exceptions.Timeout(), _FakeResponse(_SUCCESS_BODY)]

    with patch("requests.post", side_effect=responses) as post, patch("time.sleep"):
        res = engine.execute_with_retry("q")

    assert res.startswith("Page 1")
    assert post.call_count == 2
    assert engine.get_stats() == {"logical_queries": 1, "cache_hits": 0, "api_requests": 2}


def test_second_query_hits_cache_after_first():
    engine = _make_engine()

    with patch("requests.post", return_value=_FakeResponse(_SUCCESS_BODY)):
        engine.execute_with_retry("q")
    # 第二次同 query 命中缓存，不再调 API。
    with patch("requests.post") as post:
        engine.execute_with_retry("q")

    assert post.call_count == 0
    # 两次逻辑搜索；一次缓存命中；一次真实 API 请求。
    assert engine.get_stats() == {"logical_queries": 2, "cache_hits": 1, "api_requests": 1}


# --- 结构化失败状态（缺口 2）与 URL/标题/rank 旁路（缺口 3） -----------------


def test_success_outcome_carries_status_and_items():
    engine = _make_engine()
    body = {
        "organic": [
            {"title": "T1", "link": "https://a", "description": "snippet A"},
            {"title": "T2", "link": "https://b", "description": "snippet B"},
        ]
    }
    with patch("requests.post", return_value=_FakeResponse(body)):
        res = engine.execute("q")

    assert res.startswith("Page 1")
    outcome = engine.get_last_outcome()
    assert outcome.status == "success"
    assert outcome.ok is True
    # 旁路保留原始 rank / title / url / description，且不进入模型可见文本。
    assert outcome.items == [
        {"rank": 1, "title": "T1", "url": "https://a", "description": "snippet A"},
        {"rank": 2, "title": "T2", "url": "https://b", "description": "snippet B"},
    ]
    assert "https://a" not in res  # URL 只作旁路，不进摘要


def test_not_found_when_no_organic():
    engine = _make_engine()
    with patch("requests.post", return_value=_FakeResponse({})):
        res = engine.execute("q")

    assert res == "No search results found."
    assert engine.get_last_outcome().status == "not_found"


def test_not_found_when_all_descriptions_empty_keeps_sidecar():
    engine = _make_engine()
    body = {"organic": [{"title": "T", "link": "https://a", "description": ""}]}
    with patch("requests.post", return_value=_FakeResponse(body)):
        res = engine.execute("q")

    assert res == "No search results found."
    outcome = engine.get_last_outcome()
    assert outcome.status == "not_found"
    # 无摘要但旁路仍保留 title/url，与真实零结果区分。
    assert outcome.items[0]["title"] == "T"


@pytest.mark.parametrize(
    "status,kind",
    [(429, "rate_limit"), (500, "server"), (401, "auth"), (400, "invalid")],
)
def test_http_error_classified(status, kind):
    engine = _make_engine()
    with patch("requests.post", return_value=_FakeResponse(status_code=status)):
        res = engine.execute("q")

    assert res == ""
    outcome = engine.get_last_outcome()
    assert outcome.status == kind
    assert outcome.error_code == str(status)


def test_network_error_classified():
    engine = _make_engine()
    with patch("requests.post", side_effect=requests.exceptions.Timeout()):
        res = engine.execute("q")

    assert res == ""
    assert engine.get_last_outcome().status == "network"


def test_auth_fails_immediately_without_retry():
    engine = _make_engine(tool_retry_count=3)
    with patch("requests.post", return_value=_FakeResponse(status_code=401)) as post, patch("time.sleep"):
        res = engine.execute_with_retry("q")

    assert res == "Search failed for query: q"
    assert post.call_count == 1  # 鉴权错误不可重试，立即失败
    assert engine.get_stats()["api_requests"] == 1


def test_execute_still_returns_model_visible_string():
    # 兼容返回形状：execute 返回字符串，结构化结果走 get_last_outcome。
    engine = _make_engine()
    with patch("requests.post", return_value=_FakeResponse(_SUCCESS_BODY)):
        res = engine.execute("q")

    assert isinstance(res, str)
    assert res.startswith("Page 1")
    assert engine.get_last_outcome().status == "success"
