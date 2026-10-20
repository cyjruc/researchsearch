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
"""证据覆盖奖励 R_evidence 的单元测试（门禁 T05）。

对齐《第 1 周工程闭环开发方案》§4.4 与 T05：重复文档不加分，缺 gold 不除零，
截断结果不冒充已读。
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


evidence = _load("verl/utils/reward_score/evidence.py", "researchsearch_evidence")


def test_full_coverage_scores_one():
    r = evidence.compute_evidence_reward(["wiki:a", "wiki:b"], ["wiki:a", "wiki:b"])
    assert r["eligible"] is True
    assert r["score"] == pytest.approx(1.0)
    assert r["num_covered"] == 2


def test_partial_coverage_scores_fraction():
    r = evidence.compute_evidence_reward(["wiki:a", "wiki:b", "wiki:c"], ["wiki:a"])
    assert r["score"] == pytest.approx(1.0 / 3.0)
    assert r["num_covered"] == 1


def test_duplicate_visible_docs_not_double_counted():
    # 重复观察同一证据不重复给分：去重后 covered=2，而非 3。
    r = evidence.compute_evidence_reward(["wiki:a", "wiki:b"], ["wiki:a", "wiki:a", "wiki:b"])
    assert r["score"] == pytest.approx(1.0)
    assert r["num_covered"] == 2


def test_missing_gold_does_not_divide_by_zero():
    r = evidence.compute_evidence_reward([], ["wiki:a"])
    assert r["eligible"] is False
    assert r["score"] == pytest.approx(0.0)
    assert r["num_gold"] == 0


def test_non_mapped_status_disables_even_with_gold():
    r = evidence.compute_evidence_reward(["wiki:a"], ["wiki:a"], evidence_status="unmapped")
    assert r["eligible"] is False
    assert r["score"] == pytest.approx(0.0)


def test_partial_status_reported_separately():
    # partial 单独报告，不缩小分母：eligible=False，由调用方按 evidence_status 统计。
    r = evidence.compute_evidence_reward(["wiki:a"], ["wiki:a"], evidence_status="partial")
    assert r["eligible"] is False
    assert r["evidence_status"] == "partial"


def test_truncated_retrieval_hit_not_counted_as_read():
    # 后端检索命中但被截断：计入 num_retrieval_hit，不计入覆盖率分子。
    r = evidence.compute_evidence_reward(
        ["wiki:a", "wiki:b"],
        ["wiki:a"],
        retrieval_hit_doc_ids=["wiki:b", "wiki:b"],
    )
    assert r["score"] == pytest.approx(0.5)
    assert r["num_covered"] == 1
    assert r["num_retrieval_hit"] == 1


def test_dedup_preserves_order_and_filters_invalid():
    r = evidence.compute_evidence_reward(
        ["wiki:b", "wiki:a"],
        [None, "wiki:b", "", "wiki:a", "wiki:b"],
    )
    assert r["visible_doc_ids"] == ["wiki:b", "wiki:a"]
    assert r["gold_doc_ids"] == ["wiki:b", "wiki:a"]
    assert r["score"] == pytest.approx(1.0)
