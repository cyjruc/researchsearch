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
"""锁定现有分段奖励语义（门禁 T04）。

对齐《第 1 周工程闭环开发方案》§4.4：``R_existing`` 保留上游分段评分语义，
不在本周重写。测试用与 launch 脚本一致的分数（structure=0.2、final=0.1、
retrieval=0）锁定行为，并单独验证 ``retrieval_score`` 的加分逻辑。

两个奖励文件只 import 标准库，但位于 ``verl`` 包内；本机缺 ``tensordict``
导致无法 ``import verl``，故用 ``importlib`` 单文件加载，不触发包初始化。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main

# 与 train_*.sh / launch_smoke.sh 一致的分段分数。
STRUCTURE = 0.2
FINAL = 0.1
RETRIEVAL = 0.0
SCORE = 1.0

# prompt 示例自带一个 <answer>，使 extract_solution（要求 >=2 个匹配）能取到
# 模型在 assistant 之后生成的答案。
PROMPT_PREFIX = (
    "Answer the given question. For example, <answer> Beijing </answer>. Question: X?\n"
)
ASSISTANT = "<|im_start|>assistant"


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


em = _load("verl/utils/reward_score/qa_em_format.py", "qa_em_format")
f1 = _load("verl/utils/reward_score/qa_f1_format.py", "qa_f1_format")


def _valid(answer: str, *, information: str = "Doc 1(Title: t) text") -> str:
    """一条通过严格状态机的完整轨迹（含一个模型生成的 <answer>）。"""
    return (
        "<think>Need to search.</think>"
        "<search>some query</search>"
        f"<information>{information}</information>"
        "<think>Now answer.</think>"
        f"<answer>{answer}</answer>"
    )


def _invalid(answer: str) -> str:
    """不含 think/search 前缀，直接 <answer>：状态机判为非法。"""
    return f"<answer>{answer}</answer>"


def _solution(sequence: str) -> str:
    return f"{PROMPT_PREFIX}{ASSISTANT}\n{sequence}"


def _call(module, answer: str, golden, *, valid: bool, information: str = "Doc 1(Title: t) text", retrieval_score=RETRIEVAL):
    seq = _valid(answer, information=information) if valid else _invalid(answer)
    solution = _solution(seq)
    ground_truth = {"target": golden}
    return module.compute_score(
        solution,
        ground_truth,
        structure_format_score=STRUCTURE,
        final_format_score=FINAL,
        retrieval_score=retrieval_score,
        score=SCORE,
    )


# --- compute_score_em：答案与格式分段语义 -----------------------------------


@pytest.mark.parametrize("func", [em.compute_score_em, f1.compute_score_f1])
def test_correct_valid_format_scores_one(func):
    assert func(_solution(_valid("Paris")), {"target": ["Paris"]},
                 structure_format_score=STRUCTURE, final_format_score=FINAL,
                 retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(1.0)


@pytest.mark.parametrize("func", [em.compute_score_em, f1.compute_score_f1])
def test_correct_invalid_format_loses_structure(func):
    # 正确 + 格式不合法 = score - structure = 0.8
    assert func(_solution(_invalid("Paris")), {"target": ["Paris"]},
                 structure_format_score=STRUCTURE, final_format_score=FINAL,
                 retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(0.8)


def test_em_wrong_valid_format_scores_structure():
    # 错误 + 格式合法 = structure = 0.2
    assert em.compute_score_em(_solution(_valid("London")), {"target": ["Paris"]},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(0.2)


def test_f1_wrong_valid_format_scores_structure():
    # 错误 + 格式合法 = structure = 0.2
    assert f1.compute_score_f1(_solution(_valid("London")), {"target": ["Paris"]},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(0.2)


def test_em_wrong_invalid_format_scores_final():
    # 错误 + 格式不合法 = final = 0.1
    assert em.compute_score_em(_solution(_invalid("London")), {"target": ["Paris"]},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(0.1)


def test_f1_wrong_invalid_format_scores_zero_not_final():
    # 上游差异：F1 版在「错误 + 格式不合法」返回 f1=0，而不是 final_format_score=0.1。
    # 这是需如实锁定的现状（方案 §4.4 本周不重写这套规则）。
    assert f1.compute_score_f1(_solution(_invalid("London")), {"target": ["Paris"]},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(0.0)


def test_em_alias_matches_any_gold():
    golden = ["Paris", "the City of Light"]
    assert em.compute_score_em(_solution(_valid("the city of light")), {"target": golden},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(1.0)


def test_answer_none_invalid_scores_zero():
    # assistant 之后无 <answer>：extract_solution 只匹配到 prompt 示例 1 个 -> None，
    # 且状态机停留在 after_think 判非法 -> 返回 0。
    seq = "<think>x</think><search>q</search><information>d</information><think>y</think>"
    assert em.compute_score_em(_solution(seq), {"target": ["Paris"]},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(0.0)


# --- retrieval_score 加分逻辑（训练中固定为 0，这里单独验证函数行为） -----------


def test_retrieval_correct_adds_retrieval_score():
    # information 含答案字符串：retrieval_correct=True -> structure + retrieval
    solution = _solution(_valid("London", information="Doc 1(Title: t) Paris is here"))
    assert em.compute_score_em(solution, {"target": ["Paris"]},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=0.1, score=SCORE) == pytest.approx(0.3)


def test_retrieval_score_zero_makes_retrieval_neutral():
    # 训练配置 retrieval_score=0：retrieval 命中与否不改变错误+合法的得分。
    solution = _solution(_valid("London", information="Doc 1(Title: t) Paris is here"))
    assert em.compute_score_em(solution, {"target": ["Paris"]},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=0.0, score=SCORE) == pytest.approx(0.2)


# --- EM 与 F1 的判分差异（部分 token 重叠） -----------------------------------


def test_em_partial_overlap_is_wrong():
    # EM 是精确匹配：部分重叠不算对。
    assert em.compute_score_em(_solution(_valid("city")), {"target": ["the city of light"]},
                               structure_format_score=STRUCTURE, final_format_score=FINAL,
                               retrieval_score=RETRIEVAL, score=SCORE) == pytest.approx(0.2)


def test_f1_partial_overlap_scores_fraction():
    # F1 版对部分重叠返回 f1 分数本身（>0），与 EM 二值判分不同。
    score = f1.compute_score_f1(_solution(_valid("city")), {"target": ["the city of light"]},
                                structure_format_score=STRUCTURE, final_format_score=FINAL,
                                retrieval_score=RETRIEVAL, score=SCORE)
    assert 0.0 < score < 1.0
