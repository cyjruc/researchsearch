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


def _doc_title(doc: dict) -> Optional[str]:
    """从 document 提取 title（``title`` 字段优先，回退 ``contents`` 首行去引号）。

    真实 Wiki 检索器两种返回结构（见 docs/researchsearch/retrieval-contract.md §2）：
    - BM25：``{"title": "已去引号", "text": ..., "contents": ...}``。
    - Dense/FAISS：``{"id": <整数行号>, "contents": "\"Title\"\\n正文"}``（无 title 字段）。
    title 级 doc_id 需要统一取 title，本函数覆盖两种结构。
    """
    title = doc.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    contents = doc.get("contents")
    if isinstance(contents, str):
        first = contents.split("\n", 1)[0].strip()
        if len(first) >= 2 and first[0] == '"' and first[-1] == '"':
            first = first[1:-1]
        if first:
            return first
    return None


def extract_doc_ids_from_retrieval(result) -> list[str]:
    """从 Wiki 检索原始返回中提取 title 级 doc_id（``wiki:<title>``），按返回顺序保序。

    title 级文档粒度（架构 §2.1 / 第 2 周 §3.2 决策）：一篇维基文章是一个文档，
    doc_id = ``wiki:<title>``。真实检索器不返回稳定的字符串 ``document.id``
    （Dense 返回的 ``id`` 是整数行号，属于段落级 chunk，不能作为文档级 doc_id），
    因此统一从 ``document.title``（BM25 已去引号）或 ``document.contents`` 首行
    （Dense）构造 title 级 doc_id。仅当上游显式给出 ``wiki:`` 前缀的字符串
    ``document.id``（真正的稳定文档级 id）时才优先使用；段落级行号即使以字符串
    存储（真实 wiki-18 的 ``{"id": "0", ...}``）也不会被误当 doc_id，而是回退
    从 title 构造。

    Agent 观察由 ``_passages2string`` 拼成 ``Doc N(Title: ...) ...``，``document``
    里的 id 在该处被丢弃。树内证据接线需要旁路保留每个搜索实际返回的文档 ID，
    因此本函数直接从原始 ``result`` 提取 doc_id，供 :func:`compute_evidence_reward`
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
        if isinstance(doc_id, str) and doc_id.strip().startswith("wiki:"):
            ids.append(doc_id.strip())
            continue
        title = _doc_title(doc)
        if title:
            ids.append(f"wiki:{title}")
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


def combine_score_with_evidence(base_score: float, evidence_result: dict, alpha: float) -> float:
    """按 R_total = R_existing + alpha * R_evidence 合并基础奖励与证据覆盖率。

    ``alpha=0`` 时证据项完全关闭，直接返回 ``base_score``；``evidence_result``
    的 ``eligible=False`` 时其 ``score`` 为 0，合并结果等于 ``base_score``。
    本函数不判定答案对错，只做加权求和——是否只在答案正确时叠加由调用方决定。
    独立成纯函数是为了无 GPU 锁定「alpha 开关 / 非 eligible 零贡献」两条语义。
    """
    if alpha == 0.0:
        return float(base_score)
    return float(base_score) + float(alpha) * float(evidence_result.get("score", 0.0))


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


def split_doc_ids_by_truncation(
    doc_ids: Iterable[str],
    doc_token_lengths: Iterable[int],
    *,
    prefix_token_length: int = 0,
    max_obs_length: int,
) -> dict:
    """按「截断后仍完整保留」划分检索返回的文档 ID（方案 §4.2 最细部分）。

    判定口径对齐 ``generation_ts._process_next_obs``：observation 整体 tokenize 后
    只保留前 ``max_obs_length`` 个 token。文档 ``i`` 的 token 区间为
    ``[prefix + Σ_{j<i} len_j, prefix + Σ_{j≤i} len_j)``；结束位置
    ``<= max_obs_length`` 即完整保留（进入 Agent 观察），开始位置
    ``< max_obs_length`` 但结束位置超出的文档被部分截断，同样视为未进入观察。
    ``visible`` 与 ``retrieval_hit`` 互斥，并集为后端返回的全部文档，均保序、
    不去重（去重由 :func:`compute_evidence_reward` 负责）。

    「不能用后端返回了几个替代」正是本函数要防的：后端返回 topk 个文档，但被
    ``max_obs_length`` 截断后，只有完整保留的那几个真正进入 Agent 观察，其余
    只记 ``retrieval_hit``（架构 §2.1 的证据奖励与 Recall@K 分离口径）。

    Args:
        doc_ids: 检索返回的文档 ID，按检索顺序（可重复，不去重）。
        doc_token_lengths: 与 ``doc_ids`` 等长，每个文档格式化后
            （``Doc N(Title: ...) 正文\\n``）的 token 数。
        prefix_token_length: observation 前缀（``\\n\\n<information>``）的 token 数，
            作为第一个文档的起始偏移。
        max_obs_length: ``_process_next_obs`` 的截断长度（token 数）。

    Returns:
        dict：``visible_doc_ids``（截断后仍完整保留、进入观察的 doc_id）与
        ``retrieval_hit_doc_ids``（后端返回但被截断、未进入观察的 doc_id）。

    近似说明：逐文档 token 数之和与整段 observation 一次 tokenize 可能有 BPE 跨
    边界合并的 ±1~2 token 差异（每个文档以 ``\\n`` 结尾，天然是 token 边界）。
    真实接入若需严格对齐 ``_process_next_obs`` 的整段 tokenize，可用 tokenizer
    offsets 定位每个文档的结束位置，判定规则（结束位置 <= ``max_obs_length``）
    不变。
    """
    ids = list(doc_ids)
    lengths = [int(n) for n in doc_token_lengths]
    if len(ids) != len(lengths):
        raise ValueError(
            f"doc_ids 与 doc_token_lengths 长度不一致：{len(ids)} != {len(lengths)}"
        )

    visible: list[str] = []
    retrieval_hit: list[str] = []
    end = prefix_token_length
    for doc_id, length in zip(ids, lengths):
        start = end
        end = start + length
        if end <= max_obs_length:
            visible.append(doc_id)
        else:
            retrieval_hit.append(doc_id)
    return {"visible_doc_ids": visible, "retrieval_hit_doc_ids": retrieval_hit}
