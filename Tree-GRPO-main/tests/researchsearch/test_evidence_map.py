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
"""证据映射 evidence_map_fn 骨架测试（第 2 周方案 §3.2 / Day 1）。

锁定四态判定与不静默失败：

- 无支持事实 → not_available；全部可映射 → mapped；部分 → partial；全不可 → unmapped。
- 同名 title 多文档歧义：默认记 ambiguous_titles 待人工核查，不猜测。
- 语料缺失 title 记 missing_titles，不误判为可映射（架构 §2.1 防止语料缺失误判为
  搜索策略失败）。
- 多事实指向同一 doc_id 时去重。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evidence_map = _load("researchsearch/data/evidence_map.py", "researchsearch_evidence_map")

_TITLE_MAP = {
    "Paris": ["wiki:paris"],
    "Eiffel Tower": ["wiki:eiffel"],
    "Louvre": ["wiki:louvre"],
    "Ambiguous City": ["wiki:amb1", "wiki:amb2"],
}


# --- 四态判定 ------------------------------------------------------------------


def test_no_facts_is_not_available():
    result = evidence_map.map_supporting_facts([], _TITLE_MAP)
    assert result["evidence_status"] == "not_available"
    assert result["supporting_doc_ids"] == []


def test_all_mapped():
    facts = [["Paris", 0], ["Eiffel Tower", 1]]
    result = evidence_map.map_supporting_facts(facts, _TITLE_MAP)
    assert result["evidence_status"] == "mapped"
    assert result["supporting_doc_ids"] == ["wiki:paris", "wiki:eiffel"]


def test_partial_when_some_title_missing():
    facts = [["Paris", 0], ["Not In Corpus", 0]]
    result = evidence_map.map_supporting_facts(facts, _TITLE_MAP)
    assert result["evidence_status"] == "partial"
    assert result["missing_titles"] == ["Not In Corpus"]
    assert result["supporting_doc_ids"] == ["wiki:paris"]


def test_all_missing_is_unmapped():
    result = evidence_map.map_supporting_facts([["Ghost", 0], ["Other", 0]], _TITLE_MAP)
    assert result["evidence_status"] == "unmapped"
    assert result["missing_titles"] == ["Ghost", "Other"]


# --- 歧义与去重 ------------------------------------------------------------------


def test_ambiguous_title_is_recorded_not_guessed():
    facts = [["Paris", 0], ["Ambiguous City", 0]]
    result = evidence_map.map_supporting_facts(facts, _TITLE_MAP)
    assert result["evidence_status"] == "partial"
    assert result["ambiguous_titles"] == ["Ambiguous City"]
    assert result["supporting_doc_ids"] == ["wiki:paris"]


def test_ambiguous_resolver_resolves_to_mapped():
    def resolver(title, doc_ids):
        return doc_ids[0]  # 显式取第一个

    facts = [["Ambiguous City", 0]]
    result = evidence_map.map_supporting_facts(facts, _TITLE_MAP, ambiguous_resolver=resolver)
    assert result["evidence_status"] == "mapped"
    assert result["supporting_doc_ids"] == ["wiki:amb1"]


def test_duplicate_facts_dedupe_doc_ids():
    # 两个事实都指向 Paris（同 title），doc_id 只记一次。
    facts = [["Paris", 0], ["Paris", 3]]
    result = evidence_map.map_supporting_facts(facts, _TITLE_MAP)
    assert result["supporting_doc_ids"] == ["wiki:paris"]
    assert result["num_resolved"] == 2  # 两个事实都成功映射


# --- 格式兼容 ------------------------------------------------------------------


def test_dict_facts_format():
    facts = [{"title": "Paris", "sentence_index": 0}]
    result = evidence_map.map_supporting_facts(facts, _TITLE_MAP)
    assert result["supporting_doc_ids"] == ["wiki:paris"]


def test_malformed_fact_raises():
    with pytest.raises(ValueError, match="不支持的支持事实格式"):
        evidence_map.map_supporting_facts([42], _TITLE_MAP)


# --- make_evidence_map_fn 闭包 ---------------------------------------------------


def test_make_evidence_map_fn_returns_hook_shape():
    fn = evidence_map.make_evidence_map_fn(_TITLE_MAP)
    example = {"supporting_facts": [["Paris", 0], ["Eiffel Tower", 1]]}
    out = fn(example, idx=7)
    assert out["evidence_status"] == "mapped"
    assert out["supporting_doc_ids"] == ["wiki:paris", "wiki:eiffel"]


def test_make_evidence_map_fn_missing_facts_is_not_available():
    fn = evidence_map.make_evidence_map_fn(_TITLE_MAP)
    out = fn({"question": "no facts here"}, idx=0)
    assert out["evidence_status"] == "not_available"
    assert out["supporting_doc_ids"] == []


# --- FlashRAG 真实格式适配（第 2 周 §3.2 字段核对） ---------------------------


def test_flashrag_dict_facts_format():
    # 真实 supporting_facts 是 {"title": [...], "sent_id": [...]} 平行数组。
    facts = {"title": ["Paris", "Eiffel Tower"], "sent_id": [0, 1]}
    result = evidence_map.map_supporting_facts(facts, _TITLE_MAP)
    assert result["evidence_status"] == "mapped"
    assert result["supporting_doc_ids"] == ["wiki:paris", "wiki:eiffel"]


def test_flashrag_dict_facts_sent_id_missing_defaults_zero():
    facts = {"title": ["Paris", "Eiffel Tower"]}
    result = evidence_map.map_supporting_facts(facts, _TITLE_MAP)
    assert result["supporting_doc_ids"] == ["wiki:paris", "wiki:eiffel"]


def test_extract_hotpotqa_from_metadata():
    example = {"metadata": {"supporting_facts": {"title": ["Paris", "Louvre"], "sent_id": [0, 0]}}}
    assert evidence_map.extract_supporting_facts(example, "hotpotqa") == {
        "title": ["Paris", "Louvre"],
        "sent_id": [0, 0],
    }


def test_extract_musique_from_question_decomposition():
    example = {
        "metadata": {
            "question_decomposition": [
                {"support_paragraph": {"title": "Paris", "idx": 0, "is_supporting": True}},
                {"support_paragraph": {"title": "Skip", "idx": 1, "is_supporting": False}},
            ]
        }
    }
    assert evidence_map.extract_supporting_facts(example, "musique") == [["Paris", 0]]


def test_extract_nq_is_empty():
    assert evidence_map.extract_supporting_facts({"question": "q?"}, "nq") == []


def test_make_evidence_map_fn_with_data_source():
    fn = evidence_map.make_evidence_map_fn(_TITLE_MAP, data_source="hotpotqa")
    example = {"metadata": {"supporting_facts": {"title": ["Paris", "Eiffel Tower"], "sent_id": [0, 1]}}}
    out = fn(example, idx=0)
    assert out["evidence_status"] == "mapped"
    assert out["supporting_doc_ids"] == ["wiki:paris", "wiki:eiffel"]


# --- Unicode 归一化匹配（真实语料核对：wiki-18 NFD vs FlashRAG NFC） -----------


def test_unicode_nfc_fact_matches_nfd_map():
    # 真实场景：wiki-18 语料 title 是 NFD 分解形式，FlashRAG facts 是 NFC 预组合形式。
    import unicodedata

    nfc_title = "Éric Rohmer"
    nfd_title = unicodedata.normalize("NFD", nfc_title)
    title_map = {nfd_title: ["wiki:eric-rohmer"]}
    result = evidence_map.map_supporting_facts([[nfc_title, 0]], title_map)
    assert result["evidence_status"] == "mapped"
    assert result["supporting_doc_ids"] == ["wiki:eric-rohmer"]
    assert result["missing_titles"] == []


def test_unicode_nfd_fact_matches_nfc_map():
    # 反向：map 用 NFC，facts 用 NFD，同样归一化回退命中。
    import unicodedata

    nfc_title = "Owain Glyndŵr"
    nfd_title = unicodedata.normalize("NFD", nfc_title)
    title_map = {nfc_title: ["wiki:owain"]}
    result = evidence_map.map_supporting_facts([[nfd_title, 0]], title_map)
    assert result["evidence_status"] == "mapped"
    assert result["supporting_doc_ids"] == ["wiki:owain"]


def test_unicode_normalization_is_fallback_only():
    # 原样命中优先；归一化仅作回退，不影响正常 title 的精确匹配。
    title_map = {"Paris": ["wiki:paris"]}
    result = evidence_map.map_supporting_facts([["Paris", 0]], title_map)
    assert result["supporting_doc_ids"] == ["wiki:paris"]
