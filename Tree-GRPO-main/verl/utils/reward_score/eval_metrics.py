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
"""独立评测评分器与消融框架（架构 §4 / 第 2 周方案 §6.2、§7 Day 7）。

独立于训练奖励：只做纯答案 EM / F1 判定，不含 ``R_existing`` 的格式分段
（structure/final_format 只用于训练，不用于准确率评测）。架构 §3：「训练奖励、
格式有效率与纯答案指标分别记录，不能用训练奖励代替准确率」。

答案归一化与上游 ``qa_em.normalize_answer`` / ``qa_f1_format.normalize_answer``
一致（去冠词、去标点、小写、空白规整），由 ``tests/researchsearch/test_eval_metrics.py``
锁定二者对同一批样例输出一致，防止漂移。本模块只用标准库，便于无 GPU 离线验证。

提供：

- 题级 EM / F1 判定（支持多 gold 别名）。
- 题级平均（宏平均）与四项等权 Multi-Hop Avg EM。
- bootstrap 95% 置信区间。
- 消融对照（E2 vs E1、E3 vs E2），配对逐题差异可追溯。
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import Dict, Iterable, Sequence, Tuple

# 架构 §4：Multi-Hop Avg EM 固定包含四项，等权宏平均。
MULTI_HOP_DATASETS = ("hotpotqa", "2wiki", "musique", "bamboogle")


def normalize_answer(s: str) -> str:
    """答案归一化：去冠词、去标点、小写、空白规整。

    与 ``qa_em.normalize_answer`` / ``qa_f1_format.normalize_answer`` 一致
    （去冠词 a/an/the、去标点、小写、折叠空白）。
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _as_list(golden_answers) -> list:
    """把单个 gold 答案归一为列表（支持合法别名列表）。"""
    if isinstance(golden_answers, str):
        return [golden_answers]
    return list(golden_answers)


def em_match(prediction: str, golden_answers) -> int:
    """题级 EM 判定（0/1），支持多 gold 别名：任一 gold 归一化后与预测相等即命中。"""
    pred = normalize_answer(prediction)
    for gold in _as_list(golden_answers):
        if normalize_answer(gold) == pred:
            return 1
    return 0


def f1_match(prediction: str, golden_answers) -> float:
    """题级 token F1（0~1），支持多 gold 别名，取最大 F1。

    与 ``qa_f1_format.get_f1_score`` 一致：归一化后按 token 交并比算 F1，
    yes/no/noanswer 预测与 gold 不同则跳过，多 gold 取最大 F1。
    """
    pred = normalize_answer(prediction)
    best = 0.0
    for gold in _as_list(golden_answers):
        g = normalize_answer(gold)
        if pred in ("yes", "no", "noanswer") and pred != g:
            continue
        if g in ("yes", "no", "noanswer") and pred != g:
            continue
        pred_tokens = pred.split()
        gold_tokens = g.split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        best = max(best, (2 * precision * recall) / (precision + recall))
    return best


def mean_scores(scores: Iterable[float]) -> float:
    """题级平均（宏平均）；空集返回 0.0。"""
    s = list(scores)
    if not s:
        return 0.0
    return sum(s) / len(s)


def multi_hop_avg_em(
    dataset_em: Dict[str, float],
    *,
    datasets: Sequence[str] = MULTI_HOP_DATASETS,
) -> float:
    """四项等权宏平均 Multi-Hop Avg EM（架构 §4 公式）。

    ``dataset_em`` 须含 ``datasets`` 中的每一项；缺失数据集抛 ``KeyError``，
    不静默降权——架构要求四项等权，缺一项即不能声称 Multi-Hop Avg EM。
    """
    missing = [d for d in datasets if d not in dataset_em]
    if missing:
        raise KeyError(
            f"缺少数据集 EM：{missing}（Multi-Hop Avg EM 须四项等权，缺一不可）"
        )
    return mean_scores(dataset_em[d] for d in datasets)


def bootstrap_ci(
    scores: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """bootstrap 置信区间（默认 95%），返回 ``(lower, upper)``。

    对题级得分做有放回重抽样取均值，取百分位区间；``seed`` 固定保证可复现。
    """
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    tail = (1 - confidence) / 2
    lower = means[int(tail * len(means))]
    upper = means[int((1 - tail) * len(means)) - 1]
    return (lower, upper)


def pairwise_ablation(
    a_per_question: Dict[str, float],
    b_per_question: Dict[str, float],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """配对消融对照（架构 §4 消融；方案 §7 E2 vs E1、E3 vs E2）。

    两个实验按相同题目 ID 配对，计算题级均值差与逐题翻转矩阵，使结论可追溯到
    逐题结果。``a_per_question`` / ``b_per_question`` 键为题目 ID，值为该题得分
    （EM 0/1 或任意数值指标）。

    Returns:
        dict：``delta``（B−A 的题级均值差）、``num_questions``（配对题数）、
        ``a_mean`` / ``b_mean``、``confusion``（``both``/``a_win``/``b_win``/
        ``neither``，仅对 0/1 指标有意义）、``per_question``（逐题
        ``(qid, a_score, b_score, diff)``，可追溯）、``only_a`` / ``only_b``
        （只在单侧出现、未参与配对的题目 ID）。
    """
    a = {q: float(a_per_question[q]) for q in a_per_question}
    b = {q: float(b_per_question[q]) for q in b_per_question}
    qids = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    per_question = [(q, a[q], b[q], b[q] - a[q]) for q in qids]
    a_mean = mean_scores([a[q] for q in qids])
    b_mean = mean_scores([b[q] for q in qids])

    confusion = {"both": 0, "a_win": 0, "b_win": 0, "neither": 0}
    for q in qids:
        av, bv = a[q], b[q]
        if av > 0 and bv > 0:
            confusion["both"] += 1
        elif av > 0:
            confusion["a_win"] += 1
        elif bv > 0:
            confusion["b_win"] += 1
        else:
            confusion["neither"] += 1

    return {
        "label_a": label_a,
        "label_b": label_b,
        "delta": b_mean - a_mean,
        "num_questions": len(qids),
        "a_mean": a_mean,
        "b_mean": b_mean,
        "confusion": confusion,
        "per_question": per_question,
        "only_a": only_a,
        "only_b": only_b,
    }
