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
"""统一成本统计口径测试（T10 无卡部分 / 架构 §3 / 第 2 周方案 §6.3）。

锁定：

- ``CostStats`` 六口径字段完整、计数累加、跨实例 ``merge``、``reset``、线程安全。
- ``physical_new_tokens_from_tree`` 只把每个节点的**增量** token 计一次——共享前缀
  （父节点那一段）不在子节点重复计，information / padding 不计入。
- ``directory_size_bytes`` 递归统计 checkpoint 体积。
- ``post_retrieve`` 传入累加器时，逻辑搜索与外部请求（含重试）分开计数。
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, _REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cost = _load("researchsearch_cost_stats", "search_r1/search/cost_stats.py")
CostStats = cost.CostStats
COST_FIELDS = cost.COST_FIELDS
physical_new_tokens_from_tree = cost.physical_new_tokens_from_tree
directory_size_bytes = cost.directory_size_bytes

rc = _load("researchsearch_retrieval_client_coststats", "search_r1/search/retrieval_client.py")
post_retrieve = rc.post_retrieve
RetrievalError = rc.RetrievalError


class _SpyStats:
    """鸭子类型累加器：只记录调用次数，验证 post_retrieve 的计数点与口径。"""

    def __init__(self):
        self.logical = 0
        self.requests = 0

    def logical_search(self):
        self.logical += 1

    def http_request(self):
        self.requests += 1


class _FakeNode:
    """只有 ``responses_with_info_mask`` 的最小节点，供纯函数离线锁定口径。"""

    def __init__(self, mask):
        self.responses_with_info_mask = mask


class _FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


_URL = "http://retriever/retrieve"
_PAYLOAD = {"queries": ["q"], "topk": 3, "return_scores": True}
_PAD = 0


# --- 口径字段 --------------------------------------------------------------


def test_cost_fields_are_exactly_six_authoritative_fields():
    # 六口径是架构 §3 与方案 §6.3 的单一权威定义，字段名须与文档一致。
    assert set(COST_FIELDS) == {
        "physical_new_tokens",
        "logical_searches",
        "http_requests",
        "cache_hits",
        "gpu_time_seconds",
        "checkpoint_bytes",
    }


# --- CostStats 累加器 ------------------------------------------------------


def test_coststats_count_and_merge():
    s = CostStats()
    s.logical_search()
    s.logical_search(2)
    s.http_request(3)
    s.cache_hit()
    s.add_physical_new_tokens(100)
    s.add_gpu_seconds(1.5)
    s.add_checkpoint_bytes(36 * 1024**3)

    d = s.as_dict()
    assert d["logical_searches"] == 3
    assert d["http_requests"] == 3
    assert d["cache_hits"] == 1
    assert d["physical_new_tokens"] == 100
    assert d["gpu_time_seconds"] == 1.5
    assert d["checkpoint_bytes"] == 36 * 1024**3

    other = CostStats()
    other.logical_search()
    other.http_request()
    s.merge(other)
    assert s.as_dict()["logical_searches"] == 4
    assert s.as_dict()["http_requests"] == 4


def test_coststats_reset():
    s = CostStats()
    s.logical_search(5)
    s.reset()
    assert s.as_dict()["logical_searches"] == 0


def test_coststats_thread_safe():
    s = CostStats()

    def bump():
        for _ in range(1000):
            s.logical_search()

    threads = [threading.Thread(target=bump) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.as_dict()["logical_searches"] == 8000


# --- 物理新生成 token（共享前缀只计一次） ----------------------------------


def test_physical_new_tokens_counts_each_node_increment_once():
    # 树路径 root → A → B：A 生成 3 token，B 只生成自己的 2 token（不含 A 的 3）。
    # 物理新生成 token = 3 + 2 = 5；若把共享前缀重复计会是 8。
    root = _FakeNode(torch.tensor([], dtype=torch.long))
    a = _FakeNode(torch.tensor([1, 2, 3]))
    b = _FakeNode(torch.tensor([4, 5]))

    assert physical_new_tokens_from_tree([root, a, b], _PAD) == 5


def test_physical_new_tokens_skips_padding_and_information():
    # responses_with_info_mask 里 pad（=0）位置代表 padding 或 <information>（工具
    # 返回、非模型生成），均不计入物理新生成 token。
    nodes = [
        _FakeNode(torch.tensor([1, 2, 3])),  # 3 有效
        _FakeNode(torch.tensor([4, 0, 5])),  # 2 有效（中间 0 掩掉）
        _FakeNode(torch.tensor([0, 0])),     # 0 有效
    ]
    assert physical_new_tokens_from_tree(nodes, _PAD) == 5


def test_physical_new_tokens_skips_none_mask():
    # 某些节点可能没有 responses_with_info_mask（如未展开的占位节点），跳过不报错。
    assert physical_new_tokens_from_tree([_FakeNode(None)], _PAD) == 0


# --- checkpoint 体积 -------------------------------------------------------


def test_directory_size_bytes(tmp_path):
    (tmp_path / "a.txt").write_text("hello")  # 5 字节
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"\x00" * 10)  # 10 字节

    assert directory_size_bytes(str(tmp_path)) == 15


# --- post_retrieve 逻辑搜索 vs 外部请求（含重试） ---------------------------


def test_success_counts_one_logical_one_request():
    stats = _SpyStats()
    with patch("requests.post", return_value=_FakeResponse(200, {"result": []})):
        post_retrieve(_URL, _PAYLOAD, stats=stats)
    assert stats.logical == 1
    assert stats.requests == 1


def test_retry_then_success_counts_retries_as_requests():
    stats = _SpyStats()
    responses = [_FakeResponse(503), _FakeResponse(200, {"result": [["ok"]]})]

    def side_effect(*args, **kwargs):
        return responses.pop(0)

    with patch("requests.post", side_effect=side_effect), patch("time.sleep"):
        post_retrieve(_URL, _PAYLOAD, stats=stats)
    # 1 次逻辑搜索，2 次实际外部请求（1 初始 + 1 重试），二者分开。
    assert stats.logical == 1
    assert stats.requests == 2


def test_all_retries_exhausted_counts_every_attempt():
    stats = _SpyStats()
    with patch("requests.post", return_value=_FakeResponse(429)) as mock, patch("time.sleep"):
        with pytest.raises(RetrievalError):
            post_retrieve(_URL, _PAYLOAD, stats=stats)
    assert mock.call_count == 3
    assert stats.logical == 1
    assert stats.requests == 3


def test_network_errors_count_as_requests():
    stats = _SpyStats()
    import requests as _requests

    with patch("requests.post", side_effect=_requests.exceptions.ConnectionError()), patch("time.sleep"):
        with pytest.raises(RetrievalError):
            post_retrieve(_URL, _PAYLOAD, stats=stats)
    # 连接失败同样是一次「发起了外部请求」的尝试，含重试共 3 次。
    assert stats.logical == 1
    assert stats.requests == 3


def test_no_stats_is_backward_compatible():
    # 不传 stats 时行为与之前完全一致，不抛错。
    with patch("requests.post", return_value=_FakeResponse(200, {"result": []})):
        body = post_retrieve(_URL, _PAYLOAD)
    assert body == {"result": []}
