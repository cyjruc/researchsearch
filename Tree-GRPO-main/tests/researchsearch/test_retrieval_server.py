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
"""Wiki ``POST /retrieve`` 服务端契约测试（门禁 T03 服务端侧 / 方案 §4.3、§7）。

用 fake retriever 注入 endpoint，锁定 rollout 实际路径（``return_scores=true``）的契约：

- 返回形状 ``{"result": [[{"document": {...}, "score": <float>}]]}``。
- 外层 ``result`` 顺序与 ``queries`` 一一对应；重复查询各自独立、不合并。
- 空结果返回 ``[]``（不是 ``null``）；不足 ``topk`` 返回实际条数。
- ``topk`` 缺省回退到 ``config.retrieval_topk``。
- 非法输入（缺 ``queries`` / ``queries`` 非 list / 元素非 str）被 Pydantic 拒绝。
- ``document`` 结构（含 ``id``）原样透传，供 ``extract_doc_ids_from_retrieval`` 提取。

**边界说明**：真实 FAISS/BM25 索引、稳定 ``document.id`` 回查正文属于语料层缺口
（见 ``docs/researchsearch/retrieval-contract.md`` §2 缺口 1），本测试只用 fixture 锁定
「endpoint 透传机制」，不伪称真实 ``doc_id`` 已接通。真实索引就绪后把 fake retriever
换成真实检索器即可复用本文件验收。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load_retrieval_server():
    # 本地无 faiss/datasets/uvicorn/fastapi，mock 掉这些「真实服务启动才用」的重依赖。
    # transformers/torch/numpy/pydantic 本地可用，保留真实 import（加载期不实例化模型）。
    for name in ("faiss", "datasets", "uvicorn"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    fastapi = types.ModuleType("fastapi")

    class _FakeFastAPI:
        def __init__(self):
            self.routes = {}

        def post(self, path):
            def decorator(fn):
                self.routes[path] = fn
                return fn

            return decorator

    fastapi.FastAPI = _FakeFastAPI
    sys.modules["fastapi"] = fastapi

    spec = importlib.util.spec_from_file_location(
        "researchsearch_retrieval_server", _REPO / "search_r1/search/retrieval_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["researchsearch_retrieval_server"] = mod
    spec.loader.exec_module(mod)
    return mod


_server = _load_retrieval_server()
QueryRequest = _server.QueryRequest
retrieve = _server.app.routes["/retrieve"]


class FakeConfig:
    retrieval_topk = 10


class FakeRetriever:
    """复刻真实 retriever 的 ``batch_search`` 返回语义（含 return_score=False 只返回单值）。"""

    def __init__(self, results, scores=None):
        self.results = results
        self.scores = scores if scores is not None else [[0.9] * len(r) for r in results]
        self.calls = []

    def batch_search(self, query_list, num=None, return_score=False):
        self.calls.append((list(query_list), num, return_score))
        if return_score:
            return self.results, self.scores
        return self.results  # 真实 retriever 在 return_score=False 时只返回 results


def _set_retriever(results, scores=None):
    fake = FakeRetriever(results, scores)
    _server.retriever = fake
    _server.config = FakeConfig()
    return fake


def _doc(doc_id, title, text):
    return {"id": doc_id, "title": title, "text": text, "contents": f"{title}\n{text}"}


# --- 返回形状（return_scores=true） -------------------------------------------


def test_response_shape_return_scores_true():
    _set_retriever(
        [[_doc("wiki:1", "Paris", "capital of France")]],
        [[0.81]],
    )
    resp = retrieve(QueryRequest(queries=["capital of France?"], topk=3, return_scores=True))

    assert list(resp.keys()) == ["result"]
    inner = resp["result"][0]
    assert len(inner) == 1
    assert set(inner[0].keys()) == {"document", "score"}
    assert inner[0]["document"]["id"] == "wiki:1"
    assert inner[0]["score"] == 0.81


def test_outer_order_matches_queries():
    _set_retriever(
        [
            [_doc("wiki:1", "A", "body a")],
            [_doc("wiki:2", "B", "body b")],
            [_doc("wiki:3", "C", "body c")],
        ],
        [[0.9], [0.8], [0.7]],
    )
    resp = retrieve(QueryRequest(queries=["qA", "qB", "qC"], topk=3, return_scores=True))

    assert [r[0]["document"]["title"] for r in resp["result"]] == ["A", "B", "C"]
    assert [len(r) for r in resp["result"]] == [1, 1, 1]


def test_duplicate_queries_not_merged():
    _set_retriever(
        [[_doc("wiki:1", "A", "body")], [_doc("wiki:1", "A", "body")]],
        [[0.9], [0.9]],
    )
    resp = retrieve(QueryRequest(queries=["same", "same"], topk=3, return_scores=True))

    # 契约 §1：重复查询各自独立返回，不因查询相同去重合并。
    assert len(resp["result"]) == 2
    assert resp["result"][0][0]["document"]["id"] == "wiki:1"
    assert resp["result"][1][0]["document"]["id"] == "wiki:1"


def test_empty_result_returns_empty_list_not_null():
    _set_retriever([[]], [[]])
    resp = retrieve(QueryRequest(queries=["no match"], topk=3, return_scores=True))

    # 契约 §2：无结果返回 []（不是 null）。
    assert resp["result"] == [[]]
    assert resp["result"][0] is not None


def test_fewer_than_topk_returns_actual_count():
    _set_retriever([[_doc("wiki:1", "A", "body")]], [[0.9]])
    resp = retrieve(QueryRequest(queries=["q"], topk=5, return_scores=True))

    assert len(resp["result"][0]) == 1


def test_topk_falls_back_to_config_default():
    fake = _set_retriever([[]], [[]])
    retrieve(QueryRequest(queries=["q"], topk=None, return_scores=True))

    # endpoint：topk 为空时回退到 config.retrieval_topk（契约 §2 fallback 语义）。
    assert fake.calls[0][1] == FakeConfig.retrieval_topk


# --- document 透传（证据奖励的 doc_id 依赖它） ---------------------------------


def test_document_id_passthrough_for_evidence():
    # 真实 doc_id 是语料层缺口；本测试只锁定 endpoint 原样透传 document.id，
    # 证明 extract_doc_ids_from_retrieval 所需的字段不会被端点丢弃。
    _set_retriever(
        [[_doc("wiki:1", "Paris", "capital"), _doc("wiki:2", "London", "capital UK")]],
        [[0.9, 0.8]],
    )
    resp = retrieve(QueryRequest(queries=["capitals"], topk=3, return_scores=True))

    doc_ids = [item["document"]["id"] for item in resp["result"][0]]
    assert doc_ids == ["wiki:1", "wiki:2"]


# --- 非法输入（Pydantic 请求体校验） ------------------------------------------


def test_missing_queries_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(topk=3)


def test_queries_not_list_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(queries="not-a-list")


def test_queries_elements_must_be_str():
    with pytest.raises(ValidationError):
        QueryRequest(queries=[{"not": "a string"}])


# --- 已知缺陷（rollout 恒 return_scores=true，不触发） -------------------------


@pytest.mark.xfail(
    strict=True,
    reason="端点 results, scores = batch_search(...) 解包与 retriever 在 return_score=False "
    "只返回单值不兼容（已知缺陷，rollout 恒 true 不触发；修复后本测试转为 XPASS 提醒移除 xfail）",
)
def test_return_scores_false_should_return_docs_without_scores():
    _set_retriever([[_doc("wiki:1", "A", "body")]], [[0.9]])
    resp = retrieve(QueryRequest(queries=["q"], topk=3, return_scores=False))
    # 期望：return_scores=false 时内层直接是 document 列表（不带 score 包裹）。
    assert resp["result"][0][0]["title"] == "A"
