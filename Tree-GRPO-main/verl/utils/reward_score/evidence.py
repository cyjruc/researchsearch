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
"""证据覆盖奖励 R_evidence 的纯函数实现。

对齐《项目架构与实验设计》§2.1 与《第 1 周工程闭环开发方案》§4.4：

    R_evidence = |实际观察到的去重支持文档 ∩ gold| / |gold|

文档级覆盖，不声称句子级证据充分性或引用正确性。只统计截断后仍进入 Agent
观察的文档；后端检索命中但被截断、未进入观察的文档另记 ``retrieval hit``，
不计入分子。无可靠 gold（证据状态非 ``mapped``，或 gold 为空）时关闭证据项
（``eligible=False``），不除零。

本模块只用标准库，不依赖 ``verl`` / ``torch``，便于在无 GPU 环境离线验证。
"""

from __future__ import annotations

from typing import Iterable, Optional


def _dedup_preserve_order(ids: Iterable[Optional[str]]) -> list[str]:
    """按首次出现顺序去重，过滤 None / 空字符串。

    与 Recall@K 口径一致：按轨迹检索顺序累计，按稳定 doc_id 去重，保留首次
    出现顺序，不补齐、不重排。
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ids:
        if not isinstance(doc_id, str):
            continue
        doc_id = doc_id.strip()
        if not doc_id:
            continue
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def extract_doc_ids_from_retrieval(result) -> list[str]:
    """从 Wiki 检索原始返回中提取稳定 ``document.id``，按返回顺序保序。

    Wiki 后端返回形状（见 docs/researchsearch/retrieval-contract.md §4.3）：

        [{"document": {"id": "wiki:123", "contents": "标题\\n正文"}, "score": 0.8}, ...]

    Agent 观察由 ``_passages2string`` 拼成 ``Doc N(Title: ...) ...``，``document.id``
    在该处被丢弃。树内证据接线需要旁路保留每个搜索实际返回的文档 ID，因此本函数
    直接从原始 ``result`` 取 ``document.id``，供 :func:`compute_evidence_reward`
    的 ``visible_doc_ids`` / ``retrieval_hit_doc_ids`` 与 Recall@K 复用。

    Args:
        result: 单个查询的 Wiki 检索原始返回（``result[i]`` 是文档列表）。

    Returns:
        list[str]: 按返回顺序的 doc_id；缺失、非字符串或空 id 跳过，不补齐。
    """
    ids: list[str] = []
    if not isinstance(result, list):
        return ids
    for item in result:
        if not isinstance(item, dict):
            continue
        doc = item.get("document")
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("id")
        if isinstance(doc_id, str) and doc_id.strip():
            ids.append(doc_id.strip())
    return ids


def compute_evidence_reward(
    gold_doc_ids: Iterable[str],
    visible_doc_ids: Iterable[str],
    *,
    retrieval_hit_doc_ids: Optional[Iterable[str]] = None,
    evidence_status: str = "mapped",
) -> dict:
    """计算文档级证据覆盖率。

    Args:
        gold_doc_ids: 金标准支持文档 ID（parquet ``extra_info.supporting_doc_ids``）。
        visible_doc_ids: 截断后实际进入 Agent 观察的文档 ID，按检索顺序给出，
            含共享前缀与分支后的返回；可重复，函数内部去重。
        retrieval_hit_doc_ids: 后端检索命中但被截断、未进入观察的文档 ID。
            只记命中数（``num_retrieval_hit``），不计入覆盖率分子。
        evidence_status: ``mapped`` / ``partial`` / ``unmapped`` / ``not_available``。

    Returns:
        dict：
            ``score``（覆盖率 0~1，``eligible=False`` 时为 0.0）、
            ``eligible``（是否计入主证据召回）、``evidence_status``、
            ``num_gold``、``num_covered``、``num_retrieval_hit`` 以及去重后的
            三个文档 ID 列表。

    说明：``partial`` 样本 ``eligible=False``，由调用方按 ``evidence_status``
    单独报告，不通过缩小分母伪造高召回。
    """
    gold = _dedup_preserve_order(gold_doc_ids)
    visible = _dedup_preserve_order(visible_doc_ids)
    retrieval_hit = _dedup_preserve_order(retrieval_hit_doc_ids or [])

    eligible = evidence_status == "mapped" and len(gold) > 0
    if not eligible:
        return {
            "score": 0.0,
            "eligible": False,
            "evidence_status": evidence_status,
            "num_gold": len(gold),
            "num_covered": 0,
            "num_retrieval_hit": 0,
            "gold_doc_ids": gold,
            "visible_doc_ids": visible,
            "retrieval_hit_doc_ids": retrieval_hit,
        }

    gold_set = set(gold)
    covered = [doc_id for doc_id in visible if doc_id in gold_set]
    hit_gold = [doc_id for doc_id in retrieval_hit if doc_id in gold_set]

    return {
        "score": len(covered) / len(gold),
        "eligible": True,
        "evidence_status": evidence_status,
        "num_gold": len(gold),
        "num_covered": len(covered),
        "num_retrieval_hit": len(hit_gold),
        "gold_doc_ids": gold,
        "visible_doc_ids": visible,
        "retrieval_hit_doc_ids": retrieval_hit,
    }


def compute_recall_at_k(
    gold_doc_ids: Iterable[str],
    retrieved_doc_ids: Iterable[str],
    k: int,
) -> dict:
    """按检索顺序去重后取前 K 计算 Recall@K（架构 §4 检索指标口径）。

    与 :func:`compute_evidence_reward` 的区别：证据奖励只统计「截断后仍进入
    Agent 观察」的文档，而 Recall@K 是模型无关的检索召回评测指标，统计后端
    返回文档（按检索顺序去重后取前 K）对 gold 的覆盖率。

    Args:
        gold_doc_ids: 金标准支持文档 ID。
        retrieved_doc_ids: 检索返回的文档 ID，按检索顺序（可跨多次检索累计）。
        k: 截断位置，去重后取前 ``k`` 个；``k <= 0`` 视为空集。

    Returns:
        dict：``recall``（0~1，gold 为空时为 0.0）、``num_gold``、
        ``num_retrieved``（参与计算的前 K 个去重文档数）、``num_hit``、
        ``truncated_doc_ids``。
    """
    gold = _dedup_preserve_order(gold_doc_ids)
    deduped = _dedup_preserve_order(retrieved_doc_ids)
    truncated = deduped[:k] if k > 0 else []

    if not gold:
        return {
            "recall": 0.0,
            "num_gold": 0,
            "num_retrieved": len(truncated),
            "num_hit": 0,
            "truncated_doc_ids": truncated,
        }

    gold_set = set(gold)
    hit = [doc_id for doc_id in truncated if doc_id in gold_set]
    return {
        "recall": len(hit) / len(gold),
        "num_gold": len(gold),
        "num_retrieved": len(truncated),
        "num_hit": len(hit),
        "truncated_doc_ids": truncated,
    }
