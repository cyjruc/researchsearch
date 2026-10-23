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
"""证据映射 evidence_map_fn 的实现骨架（第 2 周方案 §3.2 / Day 1）。

``build.py::load_flashrag_source`` 已预留 ``evidence_map_fn(example, idx)`` 钩子，
本模块提供其实现：把 FlashRAG 各数据集的结构化支持事实（``[title, sentence_index]``
对）按冻结语料的 title→doc_id 映射表聚合为 ``supporting_doc_ids``，并判定
``evidence_status`` 四态。

本模块零重依赖（只用标准库），核心 ``map_supporting_facts`` 纯函数可在无 GPU /
无语料环境用合成 fixture 锁定口径；真实 title→doc_id 映射表待语料后注入。

四态判定（方案 §3.2）：

- 无支持事实 → ``not_available``。
- 全部事实唯一映射到 doc_id → ``mapped``。
- 部分可映射 → ``partial``。
- 全部不可映射（title 缺失或歧义未解决）→ ``unmapped``。
- 文档粒度为 **title 级**（架构 §2.1：文档级证据覆盖，不代表句子级证据充分性）：
  ``doc_id = wiki:<title>``，一篇维基文章是一个文档。wiki-18 的段落级切分（同一
  title 被拆成多个 chunk）在映射表构建阶段已归并为单一 doc_id（见
  ``scripts/researchsearch/build_title_docid_map.py``），因此真实映射表对每个 title
  只给出单元素列表，不会在这里触发歧义。
- 同名 title 多文档的歧义分支仅作防御保留：当映射表被手工注入多元素值时，默认不
  猜测，记入 ``ambiguous_titles`` 待人工核查；可注入 ``ambiguous_resolver(title,
  doc_ids)`` 显式解决。

映射失败不静默：``missing_titles``（语料缺失）单独返回，供 §3.2 的小样本报告逐条
核对，避免把语料缺失误判为搜索策略失败（架构 §2.1）。
"""

from __future__ import annotations

import unicodedata
from typing import Callable, Iterable, Mapping, Optional, Sequence

EVIDENCE_STATUSES = ("mapped", "partial", "unmapped", "not_available")


def _parse_facts(supporting_facts) -> list[tuple[str, int]]:
    """把支持事实归一为 ``[(title, sentence_index), ...]``。

    兼容三种格式：
    - ``[[title, sentence_index], ...]``（列表的列表）；
    - ``[{"title": ..., "sentence_index": ...}, ...]``；
    - FlashRAG 平行数组 ``{"title": [...], "sent_id": [...]}``（title 与
      sent_id 一一对应；sent_id 缺失或不足时按 0 补齐）。
    """
    if not supporting_facts:
        return []
    # FlashRAG 平行数组格式：{"title": [...], "sent_id": [...]}
    if isinstance(supporting_facts, Mapping) and isinstance(supporting_facts.get("title"), (list, tuple)):
        titles = supporting_facts["title"]
        sent_ids = supporting_facts.get("sent_id") or supporting_facts.get("sentence_index") or []
        facts: list[tuple[str, int]] = []
        for i, title in enumerate(titles):
            if not isinstance(title, str) or not title:
                raise ValueError(f"支持事实缺少非空 title：{title!r}")
            facts.append((title, int(sent_ids[i]) if i < len(sent_ids) else 0))
        return facts
    facts: list[tuple[str, int]] = []
    for fact in supporting_facts:
        if isinstance(fact, Mapping):
            title = fact.get("title")
            idx = fact.get("sentence_index", fact.get("sentence_idx", 0))
        elif isinstance(fact, (list, tuple)) and len(fact) >= 1:
            title = fact[0]
            idx = fact[1] if len(fact) >= 2 else 0
        else:
            raise ValueError(f"不支持的支持事实格式：{fact!r}")
        if not isinstance(title, str) or not title:
            raise ValueError(f"支持事实缺少非空 title：{fact!r}")
        facts.append((title, int(idx)))
    return facts


def _resolve_title(title_to_doc_ids: Mapping[str, Sequence[str]], title: str):
    """按原样、NFD、NFC 三种形式查映射表，覆盖语料与数据集的 Unicode 归一化差异。

    真实语料核对（第 2 周 §3.2）发现：wiki-18 语料的 title 以 NFD 分解形式存储
    （如 ``E + U+0301``），而 FlashRAG 支持事实以 NFC 预组合形式给出（如
    ``U+00C9``）。两者字符串不等价，直接查会误判为语料缺失。本函数先按原样查，
    再依次按 NFD / NFC 归一化后查，命中即返回，避免变音符号 title 被漏映射。
    """
    candidates = title_to_doc_ids.get(title)
    if candidates:
        return candidates
    for norm_title in (unicodedata.normalize("NFD", title), unicodedata.normalize("NFC", title)):
        if norm_title != title:
            candidates = title_to_doc_ids.get(norm_title)
            if candidates:
                return candidates
    return None


def map_supporting_facts(
    supporting_facts,
    title_to_doc_ids: Mapping[str, Sequence[str]],
    *,
    ambiguous_resolver: Optional[Callable[[str, Sequence[str]], Optional[str]]] = None,
) -> dict:
    """核心纯函数：支持事实 → 去重 ``supporting_doc_ids`` + 四态 ``evidence_status``。

    Args:
        supporting_facts: FlashRAG 支持事实（见 :func:`_parse_facts`）。
        title_to_doc_ids: 冻结语料的 ``{title: [doc_id, ...]}`` 映射表（真实语料注入；
            同名 title 可能映射多个 doc_id，即歧义）。
        ambiguous_resolver: 可选 ``(title, doc_ids) -> doc_id | None``，用于显式解决
            同名 title 歧义；返回 None 表示仍视为歧义（需人工核查）。

    Returns:
        dict：``supporting_doc_ids``（去重、保序）、``evidence_status``、
        ``ambiguous_titles``（同名多文档、待人工核查）、``missing_titles``
        （映射表缺失的 title）、``num_facts`` / ``num_resolved``。
    """
    facts = _parse_facts(supporting_facts)
    if not facts:
        return {
            "supporting_doc_ids": [],
            "evidence_status": "not_available",
            "ambiguous_titles": [],
            "missing_titles": [],
            "num_facts": 0,
            "num_resolved": 0,
        }

    doc_ids: list[str] = []
    ambiguous_titles: list[str] = []
    missing_titles: list[str] = []
    seen: set[str] = set()
    num_resolved = 0

    for title, _idx in facts:
        candidates = _resolve_title(title_to_doc_ids, title)
        if not candidates:
            missing_titles.append(title)
            continue
        if len(candidates) == 1:
            doc_id = candidates[0]
        else:
            doc_id = ambiguous_resolver(title, candidates) if ambiguous_resolver else None
            if doc_id is None:
                ambiguous_titles.append(title)
                continue
        num_resolved += 1
        if doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)

    if num_resolved == len(facts):
        status = "mapped"
    elif num_resolved == 0:
        status = "unmapped"
    else:
        status = "partial"

    return {
        "supporting_doc_ids": doc_ids,
        "evidence_status": status,
        "ambiguous_titles": ambiguous_titles,
        "missing_titles": missing_titles,
        "num_facts": len(facts),
        "num_resolved": num_resolved,
    }


def extract_supporting_facts(example, data_source: Optional[str] = None) -> list:
    """从 FlashRAG 样本提取支持事实，交给 :func:`map_supporting_facts` 解析。

    各数据源的证据字段位置不同（真实结构见第 2 周方案 §3.2 核对）：

    - ``hotpotqa`` / ``2wikimultihopqa``：``metadata.supporting_facts``
      （``{"title": [...], "sent_id": [...]}`` 平行数组）。
    - ``musique``：``metadata.question_decomposition`` 里 ``is_supporting=True``
      的 ``support_paragraph``（取 ``title`` 与 ``idx``）。
    - 其他（如 ``nq`` 单跳）：无文档级证据 → 空列表。

    ``data_source=None`` 时回退到顶层 ``example["supporting_facts"]``，保持对
    旧调用方与测试的兼容。
    """
    md = example.get("metadata") if isinstance(example, Mapping) else {}
    if data_source in ("hotpotqa", "2wikimultihopqa"):
        return md.get("supporting_facts") or []
    if data_source == "musique":
        facts: list[list] = []
        for sub in md.get("question_decomposition") or []:
            sp = sub.get("support_paragraph") or {}
            if sp.get("is_supporting") and sp.get("title"):
                facts.append([sp["title"], sp.get("idx", sp.get("paragraph_support_idx", 0))])
        return facts
    return example.get("supporting_facts") or []


def make_evidence_map_fn(
    title_to_doc_ids: Mapping[str, Sequence[str]],
    *,
    data_source: Optional[str] = None,
    ambiguous_resolver: Optional[Callable[[str, Sequence[str]], Optional[str]]] = None,
) -> Callable:
    """构造 ``build.py::load_flashrag_source`` 需要的 ``evidence_map_fn(example, idx)``。

    ``data_source`` 决定从样本哪里提取支持事实（见 :func:`extract_supporting_facts`）；
    提取后调 :func:`map_supporting_facts`，返回 ``{"supporting_doc_ids": [...],
    "evidence_status": "..."}``（与 build.py 钩子约定的返回结构一致）。``idx``
    保留参数位但当前不参与映射，便于对齐钩子签名。
    """

    def evidence_map_fn(example, idx):
        facts = extract_supporting_facts(example, data_source)
        result = map_supporting_facts(
            facts, title_to_doc_ids, ambiguous_resolver=ambiguous_resolver
        )
        return {
            "supporting_doc_ids": result["supporting_doc_ids"],
            "evidence_status": result["evidence_status"],
        }

    return evidence_map_fn
