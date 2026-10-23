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
"""独立评测评分器与消融框架测试（第 2 周方案 §6.2 / §7 Day 7）。

锁定：

- 答案归一化与上游 ``qa_em`` / ``qa_f1_format`` 输出一致（防漂移）。
- 题级 EM / F1 判定支持多 gold 别名、大小写/标点/冠词不敏感。
- 宏平均与四项等权 Multi-Hop Avg EM（缺失数据集报错，不静默降权）。
- bootstrap 置信区间可复现。
- 消融对照逐题可追溯，题目集不一致时如实暴露 ``only_a`` / ``only_b``。
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


em_metrics = _load("verl/utils/reward_score/eval_metrics.py", "researchsearch_eval_metrics")
qa_em = _load("verl/utils/reward_score/qa_em.py", "researchsearch_qa_em")
qa_f1 = _load("verl/utils/reward_score/qa_f1_format.py", "researchsearch_qa_f1")


# --- 归一化与上游一致（防漂移） --------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "The Cat",
        "A. B.",
        "  Hello   World!  ",
        "yes",
        "3.14",
        "U.S.A.",
        "don't",
        "New York City",
        "",
    ],
)
def test_normalize_answer_matches_upstream(raw):
    # 架构 §3：评测评分器复用训练奖励同一套归一化，防止答案判定漂移。
    assert em_metrics.normalize_answer(raw) == qa_em.normalize_answer(raw)
    assert em_metrics.normalize_answer(raw) == qa_f1.normalize_answer(raw)


# --- 题级 EM -----------------------------------------------------------------


def test_em_exact_match():
    assert em_metrics.em_match("paris", "Paris") == 1


def test_em_supports_aliases():
    # 多 gold 别名：任一命中即 1。
    assert em_metrics.em_match("USA", ["United States", "U.S.A.", "America"]) == 1


def test_em_normalizes_case_punct_articles():
    assert em_metrics.em_match("The Eiffel Tower!", "the eiffel tower") == 1


def test_em_no_match_is_zero():
    assert em_metrics.em_match("London", "Paris") == 0


# --- 题级 F1 -----------------------------------------------------------------


def test_f1_perfect_match_is_one():
    assert em_metrics.f1_match("the cat", "a cat") == pytest.approx(1.0)


def test_f1_partial_overlap():
    # pred tokens [x,y,z] vs gold [x,y]：precision 2/3、recall 1 → F1 0.8。
    # （不用 a/b/c 作样例：a 是冠词会被归一化去掉，见 test_normalize_answer_matches_upstream。）
    assert em_metrics.f1_match("x y z", "x y") == pytest.approx(0.8)


def test_f1_takes_max_over_aliases():
    # 别名取最大 F1：第一个 gold 完全匹配。
    assert em_metrics.f1_match("cat", ["cat", "dog", "bird"]) == pytest.approx(1.0)


def test_f1_yes_no_mismatch_is_zero():
    # yes/no 预测与 gold 不同则跳过，F1=0（与 qa_f1_format.get_f1_score 一致）。
    assert em_metrics.f1_match("yes", "no") == pytest.approx(0.0)


# --- 宏平均 -----------------------------------------------------------------


def test_mean_scores_empty_is_zero():
    assert em_metrics.mean_scores([]) == pytest.approx(0.0)


def test_mean_scores_average():
    assert em_metrics.mean_scores([0.0, 1.0, 1.0, 0.0]) == pytest.approx(0.5)


def test_multi_hop_avg_em_equal_weight():
    # 四项等权：平均，不按样本量加权。
    dataset_em = {"hotpotqa": 0.4, "2wiki": 0.5, "musique": 0.6, "bamboogle": 0.3}
    assert em_metrics.multi_hop_avg_em(dataset_em) == pytest.approx((0.4 + 0.5 + 0.6 + 0.3) / 4)


def test_multi_hop_avg_em_missing_dataset_raises():
    # 缺一项即不能声称 Multi-Hop Avg EM，报错而非静默降权。
    with pytest.raises(KeyError):
        em_metrics.multi_hop_avg_em({"hotpotqa": 0.4, "2wiki": 0.5, "musique": 0.6})


# --- bootstrap 置信区间 -----------------------------------------------------


def test_bootstrap_ci_deterministic():
    scores = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    assert em_metrics.bootstrap_ci(scores, seed=7) == em_metrics.bootstrap_ci(scores, seed=7)


def test_bootstrap_ci_all_one_is_point_mass():
    lo, hi = em_metrics.bootstrap_ci([1.0] * 10, seed=0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_empty():
    assert em_metrics.bootstrap_ci([]) == (0.0, 0.0)


# --- 消融对照 ---------------------------------------------------------------


def test_ablation_delta_and_per_question():
    a = {"q1": 0, "q2": 1, "q3": 0}
    b = {"q1": 0, "q2": 1, "q3": 1}
    r = em_metrics.pairwise_ablation(a, b, label_a="E1", label_b="E2")
    assert r["num_questions"] == 3
    assert r["a_mean"] == pytest.approx(1 / 3)
    assert r["b_mean"] == pytest.approx(2 / 3)
    assert r["delta"] == pytest.approx(1 / 3)
    # 逐题可追溯：(qid, a_score, b_score, diff)
    assert dict((q, d) for q, _, _, d in r["per_question"]) == {"q1": 0, "q2": 0, "q3": 1}


def test_ablation_confusion_matrix():
    a = {"q1": 0, "q2": 1, "q3": 1, "q4": 0}
    b = {"q1": 1, "q2": 1, "q3": 0, "q4": 0}
    r = em_metrics.pairwise_ablation(a, b)
    assert r["confusion"] == {"both": 1, "a_win": 1, "b_win": 1, "neither": 1}


def test_ablation_mismatched_question_sets_reported():
    # 题目集不一致：只配对交集，并如实暴露单侧题目，不静默。
    a = {"q1": 0, "q2": 1, "only_a_q": 1}
    b = {"q1": 1, "q2": 1, "only_b_q": 0}
    r = em_metrics.pairwise_ablation(a, b)
    assert r["num_questions"] == 2
    assert r["only_a"] == ["only_a_q"]
    assert r["only_b"] == ["only_b_q"]
