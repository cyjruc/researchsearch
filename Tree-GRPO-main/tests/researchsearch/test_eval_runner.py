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
"""评测 runner 核心测试（第 2 周方案 §7 Day 7 / 门禁 E06、E07）。

锁定：

- 逐题预测 schema 校验（缺字段 / question_id 重复 / 字段类型错误不静默）。
- EM 独立于训练奖励：不读任何 reward / format 字段。
- Recall@K 与证据集合分离：Recall@K 读 retrieved_doc_ids，不读 visible_doc_ids。
- 证据可映射比例分母 = mapped + partial + unmapped（not_available 不进分母）。
- Multi-Hop Avg EM 四项等权，缺数据集如实标注 missing 而非静默沿用前三项。
- 消融按 question_id 配对，题目集不一致暴露 only_a / only_b。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("verl/utils/reward_score/eval_runner.py", "researchsearch_eval_runner")
em_metrics = _load("verl/utils/reward_score/eval_metrics.py", "researchsearch_eval_metrics")
evidence = _load("verl/utils/reward_score/evidence.py", "researchsearch_evidence")


def _scorers() -> SimpleNamespace:
    return SimpleNamespace(
        em_match=em_metrics.em_match,
        f1_match=em_metrics.f1_match,
        mean_scores=em_metrics.mean_scores,
        multi_hop_avg_em=em_metrics.multi_hop_avg_em,
        bootstrap_ci=em_metrics.bootstrap_ci,
        pairwise_ablation=em_metrics.pairwise_ablation,
        compute_recall_at_k=evidence.compute_recall_at_k,
    )


def _pred(
    qid,
    dataset,
    prediction,
    gold,
    retrieved=None,
    visible=None,
    gold_docs=None,
    status=None,
    **extra,
):
    item = {
        "question_id": qid,
        "dataset": dataset,
        "prediction": prediction,
        "golden_answers": gold,
    }
    if retrieved is not None:
        item["retrieved_doc_ids"] = retrieved
    if visible is not None:
        item["visible_doc_ids"] = visible
    if gold_docs is not None:
        item["gold_doc_ids"] = gold_docs
    if status is not None:
        item["evidence_status"] = status
    item.update(extra)
    return item


# --- schema 校验 ---------------------------------------------------------------


def test_validate_missing_required_field_raises():
    with pytest.raises(ValueError):
        runner.validate_predictions(
            [{"dataset": "hotpotqa", "prediction": "x", "golden_answers": "x"}]
        )


def test_validate_duplicate_question_id_raises():
    with pytest.raises(ValueError, match="question_id 重复"):
        runner.validate_predictions(
            [_pred("q1", "hotpotqa", "a", "a"), _pred("q1", "hotpotqa", "b", "b")]
        )


def test_validate_gold_string_becomes_list_and_status_defaults():
    out = runner.validate_predictions([_pred("q1", "hotpotqa", "Paris", "Paris")])
    assert out[0]["golden_answers"] == ["Paris"]
    assert out[0]["evidence_status"] == "not_available"
    assert out[0]["retrieved_doc_ids"] == []


def test_validate_bad_evidence_status_raises():
    with pytest.raises(ValueError, match="evidence_status"):
        runner.validate_predictions(
            [_pred("q1", "hotpotqa", "a", "a", status="unknown")]
        )


def test_validate_id_list_must_be_list():
    with pytest.raises(ValueError, match="retrieved_doc_ids"):
        runner.validate_predictions(
            [_pred("q1", "hotpotqa", "a", "a", retrieved="wiki:1")]
        )


# --- EM 独立于训练奖励 ---------------------------------------------------------


def test_em_ignores_reward_and_format_fields():
    # 额外塞入 reward / format 字段不影响 EM（门禁 E06：EM 独立于训练奖励）。
    base = _pred("q1", "hotpotqa", "Paris", "Paris")
    noisy = _pred(
        "q1",
        "hotpotqa",
        "Paris",
        "Paris",
        reward=0.5,
        format_score=0.0,
        is_valid=False,
    )
    report_base = runner.evaluate_predictions([base], scorers=_scorers())
    report_noisy = runner.evaluate_predictions([noisy], scorers=_scorers())
    assert report_base["em"]["overall"] == 1.0
    assert report_noisy["em"]["overall"] == 1.0


# --- Recall@K 与证据集合分离 ---------------------------------------------------


def test_recall_uses_retrieved_not_visible():
    # retrieved 命中 gold，但 visible 不命中：Recall@K 应为 1（读检索返回集合）。
    p = _pred(
        "q1",
        "hotpotqa",
        "a",
        "a",
        retrieved=["wiki:1", "wiki:2"],
        visible=["wiki:99"],
        gold_docs=["wiki:1"],
        status="mapped",
    )
    report = runner.evaluate_predictions([p], scorers=_scorers(), k_values=[5])
    assert report["recall"][5]["hotpotqa"]["recall"] == pytest.approx(1.0)


def test_recall_does_not_use_visible_when_retrieved_misses():
    # 反例：visible 命中但 retrieved 不命中，Recall@K 应为 0（两个集合不混用）。
    p = _pred(
        "q1",
        "hotpotqa",
        "a",
        "a",
        retrieved=["wiki:98"],
        visible=["wiki:1"],
        gold_docs=["wiki:1"],
        status="mapped",
    )
    report = runner.evaluate_predictions([p], scorers=_scorers(), k_values=[5])
    assert report["recall"][5]["hotpotqa"]["recall"] == pytest.approx(0.0)


def test_recall_skips_samples_without_gold_docs():
    # 无 gold_doc_ids 的样本不参与 Recall@K 平均（num_annotated 只计有标注的）。
    p1 = _pred("q1", "hotpotqa", "a", "a", retrieved=["wiki:1"], gold_docs=["wiki:1"])
    p2 = _pred("q2", "hotpotqa", "b", "b", retrieved=["wiki:1"])  # 无 gold_docs
    report = runner.evaluate_predictions([p1, p2], scorers=_scorers(), k_values=[5])
    assert report["recall"][5]["hotpotqa"]["num_annotated"] == 1
    assert report["recall"][5]["hotpotqa"]["recall"] == pytest.approx(1.0)


# --- 证据可映射比例 -------------------------------------------------------------


def test_mappable_ratio_denominator_excludes_not_available():
    preds = [
        _pred("q1", "hotpotqa", "a", "a", status="mapped"),
        _pred("q2", "hotpotqa", "b", "b", status="mapped"),
        _pred("q3", "hotpotqa", "c", "c", status="partial"),
        _pred("q4", "hotpotqa", "d", "d", status="not_available"),
    ]
    result = runner.compute_mappable_ratio(preds)
    # 分母 = mapped + partial + unmapped = 3，不含 not_available。
    assert result["annotated"] == 3
    assert result["ratio"] == pytest.approx(2 / 3)


def test_mappable_ratio_empty_annotated_is_zero():
    preds = [_pred("q1", "hotpotqa", "a", "a", status="not_available")]
    result = runner.compute_mappable_ratio(preds)
    assert result["annotated"] == 0
    assert result["ratio"] == 0.0


# --- 完整评测报告 ---------------------------------------------------------------


def test_evaluate_multi_hop_avg_when_all_four_present():
    preds = [
        _pred("h1", "hotpotqa", "paris", "Paris"),
        _pred("w1", "2wiki", "tokyo", "Tokyo"),
        _pred("m1", "musique", "x", "y"),
        _pred("b1", "bamboogle", "x", "y"),
    ]
    report = runner.evaluate_predictions(preds, scorers=_scorers())
    # hotpotqa/2wiki 答对，musique/bamboogle 答错 → 四项等权 (1+1+0+0)/4。
    assert report["multi_hop_avg_em"] == pytest.approx(0.5)
    assert report["multi_hop_missing"] == []


def test_evaluate_multi_hop_avg_missing_is_reported_not_silent():
    preds = [_pred("h1", "hotpotqa", "paris", "Paris")]
    report = runner.evaluate_predictions(preds, scorers=_scorers())
    assert report["multi_hop_avg_em"] is None
    assert report["multi_hop_missing"] == ["2wiki", "musique", "bamboogle"]


def test_evaluate_em_ci_available():
    preds = [_pred("h1", "hotpotqa", "paris", "Paris"), _pred("h2", "hotpotqa", "x", "Paris")]
    report = runner.evaluate_predictions(preds, scorers=_scorers())
    lo, hi = report["em"]["ci95"]
    assert 0.0 <= lo <= hi <= 1.0


# --- 消融 ----------------------------------------------------------------------


def test_ablation_pairs_by_question_id_and_reports_only_sides():
    a = [_pred("q1", "hotpotqa", "a", "a"), _pred("q2", "hotpotqa", "b", "b"), _pred("only_a", "hotpotqa", "x", "x")]
    b = [_pred("q1", "hotpotqa", "a", "a"), _pred("q2", "hotpotqa", "c", "b"), _pred("only_b", "hotpotqa", "y", "y")]
    result = runner.run_ablation(a, b, scorers=_scorers(), label_a="E1", label_b="E2")
    assert result["num_questions"] == 2
    assert result["only_a"] == ["only_a"]
    assert result["only_b"] == ["only_b"]
    assert result["label_a"] == "E1"
    assert result["label_b"] == "E2"


def test_ablation_rejects_unknown_metric():
    with pytest.raises(ValueError, match="metric"):
        runner.run_ablation([], [], scorers=_scorers(), metric="reward")
