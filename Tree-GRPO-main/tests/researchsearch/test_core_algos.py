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
"""分组优势计算测试（门禁 T07）。

对齐《第 1 周工程闭环开发方案》Day 5 任务 3 / T07：用固定数值验证零方差、
单元素及混合奖励组；同题分组正确、树内/树间不串组、零方差有限（不产生
NaN/inf）。

本文件只测 ``core_algos`` 三个纯 torch 函数，不改算法。核心事实：

- 默认 ``adv_estimator='tree'`` 是两次 ``compute_grpo_outcome_advantage``
  （按题目 ``uid``、按树 ``tree_uid``）**相加**，不是 ``tree_2norm`` 路由的
  ``compute_tree_grpo_outcome_advantage``（顺序双重标准化）。
- 优势标准化使用 torch 的样本标准差（``torch.std`` 默认 unbiased，除以 n-1）。

本机缺 ``tensordict`` 导致无法 ``import verl``；``core_algos.py`` 只在本文件
三个函数之外用到 ``verl_F``，故用占位模块绕过包初始化后单文件加载。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load_core_algos():
    # 绕过 verl 包初始化（本机缺 tensordict）：三个被测函数不使用 verl_F。
    for name in ("verl", "verl.utils"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    if "verl.utils.torch_functional" not in sys.modules:
        sys.modules["verl.utils.torch_functional"] = types.ModuleType("verl.utils.torch_functional")
    spec = importlib.util.spec_from_file_location(
        "researchsearch_core_algos", _REPO / "verl/trainer/ppo/core_algos.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load_core_algos()

INV_SQRT2 = 1.0 / math.sqrt(2.0)
SQRT2 = math.sqrt(2.0)


# --- compute_grpo_outcome_advantage ------------------------------------------


def test_outcome_mixed_reward_and_zero_variance_groups():
    # 两个题目（index 0/1），各 2 个样本。index=0 组奖励有差异；index=1 组全同。
    # 奖励只在最后一个 token（EOS 位置）为标量。
    rewards = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # index 0
            [0.0, 0.0, 3.0],  # index 0
            [0.0, 0.0, 2.0],  # index 1
            [0.0, 0.0, 2.0],  # index 1
        ]
    )
    eos_mask = torch.ones(4, 3)
    index = [0, 0, 1, 1]

    adv, ret = ca.compute_grpo_outcome_advantage(rewards, eos_mask, index)

    # index=0 组 [1,3]：mean=2，样本标准差 sqrt(2) -> +-1/sqrt2。
    # index=1 组 [2,2]：std=0 -> 标准化后为 0（有限，非 NaN）。
    expected = torch.tensor(
        [
            [-INV_SQRT2, -INV_SQRT2, -INV_SQRT2],
            [INV_SQRT2, INV_SQRT2, INV_SQRT2],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(adv, expected, atol=1e-5)
    # outcome 版返回的优势与 returns 相同。
    assert torch.allclose(ret, expected, atol=1e-5)
    # 零方差组必须有限。
    assert torch.isfinite(adv).all()


def test_outcome_single_element_group_not_normalized():
    # 单元素组：mean=0、std=1 -> 优势 = 原始分数（仅除以 1，不减均值）。
    rewards = torch.tensor([[0.0, 0.0, 2.5]])
    eos_mask = torch.ones(1, 3)
    adv, _ = ca.compute_grpo_outcome_advantage(rewards, eos_mask, [7])
    assert torch.allclose(adv, torch.tensor([[2.5, 2.5, 2.5]]), atol=1e-6)


def test_outcome_eos_mask_zeroes_advantage():
    # 标准化值只落在 eos_mask=1 的位置，mask=0 的位置优势为 0。
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    eos_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    adv, _ = ca.compute_grpo_outcome_advantage(rewards, eos_mask, [0, 0])
    expected = torch.tensor(
        [[-INV_SQRT2, -INV_SQRT2, 0.0], [INV_SQRT2, INV_SQRT2, 0.0]]
    )
    assert torch.allclose(adv, expected, atol=1e-5)


def test_outcome_groups_do_not_cross_contaminate():
    # 两个不同题目即使原始行号相同（index 全 0）也应各自独立标准化；
    # 这里用两个不同 index 组验证：每组只按组内分数标准化。
    rewards = torch.tensor([[0.0, 1.0], [0.0, 3.0], [0.0, 100.0], [0.0, 102.0]])
    eos_mask = torch.ones(4, 2)
    index = [0, 0, 1, 1]
    adv, _ = ca.compute_grpo_outcome_advantage(rewards, eos_mask, index)
    # index=0 组 [1,3] -> +-1/sqrt2；index=1 组 [100,102] -> +-1/sqrt2（独立）。
    expected = torch.tensor(
        [
            [-INV_SQRT2, -INV_SQRT2],
            [INV_SQRT2, INV_SQRT2],
            [-INV_SQRT2, -INV_SQRT2],
            [INV_SQRT2, INV_SQRT2],
        ]
    )
    assert torch.allclose(adv, expected, atol=1e-5)


def test_outcome_accepts_numpy_index_like_trainer():
    # 实际调用 index 来自 non_tensor_batch（numpy 数组），行为与 list 一致。
    import numpy as np

    rewards = torch.tensor([[0.0, 1.0], [0.0, 3.0]])
    eos_mask = torch.ones(2, 2)
    adv, _ = ca.compute_grpo_outcome_advantage(rewards, eos_mask, np.array([0, 0]))
    expected = torch.tensor([[-INV_SQRT2, -INV_SQRT2], [INV_SQRT2, INV_SQRT2]])
    assert torch.allclose(adv, expected, atol=1e-5)


# --- compute_tree_grpo_outcome_advantage（tree_2norm 路由） -------------------


def test_tree_2norm_is_sequential_double_normalization():
    # 顺序双重标准化：先按 index 标准化，再按 tree_index 对标准化结果再标准化。
    rewards = torch.tensor([[1.0], [3.0], [10.0], [30.0]])
    eos_mask = torch.ones(4, 1)
    index = [0, 0, 1, 1]      # 题目分组：样本 0,1 一组；2,3 一组
    tree_index = [0, 1, 1, 0]  # 树分组：样本 0,3 一树；1,2 一树

    adv, _ = ca.compute_tree_grpo_outcome_advantage(rewards, eos_mask, index, tree_index)

    # 第一层（index）：
    #   [1,3] -> mean=2, std=sqrt(2) -> -1/sqrt2, +1/sqrt2
    #   [10,30] -> mean=20, std=10*sqrt(2) -> -1/sqrt2, +1/sqrt2
    # 第二层（tree）：[-,+] 与 [+,-] 的 mean=0, std=1 -> 保持 +-1/sqrt2。
    expected = torch.tensor([[-INV_SQRT2], [INV_SQRT2], [-INV_SQRT2], [INV_SQRT2]])
    assert torch.allclose(adv, expected, atol=1e-5)
    assert torch.isfinite(adv).all()


def test_tree_default_adds_two_independent_normalizations_not_tree_2norm():
    # 默认 tree = 按 uid 标准化 + 按 tree_uid 标准化（相加），与 tree_2norm 不同。
    rewards = torch.tensor([[1.0], [3.0], [10.0], [30.0]])
    eos_mask = torch.ones(4, 1)
    uid = [0, 0, 1, 1]
    tree_uid = [0, 1, 1, 0]

    inter, _ = ca.compute_grpo_outcome_advantage(rewards, eos_mask, uid)
    inner, _ = ca.compute_grpo_outcome_advantage(rewards, eos_mask, tree_uid)
    combined = inter + inner

    # inter（按题目）：[-1/sqrt2, +1/sqrt2, -1/sqrt2, +1/sqrt2]
    # inner（按树）：
    #   tree 0: [1,30] -> mean=15.5, std=29/sqrt2 -> -1/sqrt2, +1/sqrt2
    #   tree 1: [3,10] -> mean=6.5,  std=7/sqrt2  -> -1/sqrt2, +1/sqrt2
    #   inner = [-1/sqrt2, -1/sqrt2, +1/sqrt2, +1/sqrt2]
    # combined = [-sqrt2, 0, 0, +sqrt2]
    expected = torch.tensor([[-SQRT2], [0.0], [0.0], [SQRT2]])
    assert torch.allclose(combined, expected, atol=1e-5)

    # 与 tree_2norm 明显不同（防止误用非默认路由）。
    tree2, _ = ca.compute_tree_grpo_outcome_advantage(rewards, eos_mask, uid, tree_uid)
    assert not torch.allclose(combined, tree2, atol=1e-3)


# --- compute_grpo_per_token_advantage（tree_per_token 路由） -------------------


def test_per_token_pools_valid_tokens_across_group():
    # 组内所有有效 token 汇集后做单一标准化。
    rewards = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    eos_mask = torch.ones(2, 2)
    adv, _ = ca.compute_grpo_per_token_advantage(rewards, eos_mask, [0, 0])

    # pool=[0,1,2,3]，mean=1.5，样本标准差 sqrt(5/3)。
    std = math.sqrt(5.0 / 3.0)
    expected = torch.tensor(
        [
            [(0.0 - 1.5) / std, (1.0 - 1.5) / std],
            [(2.0 - 1.5) / std, (3.0 - 1.5) / std],
        ]
    )
    assert torch.allclose(adv, expected, atol=1e-5)


def test_per_token_single_element_pool_zeroes_itself():
    # 单 token pool（numel==1）：mean 设为该 token 本身 -> (x-x)/1 = 0。
    # 注意与 outcome 版单元素（mean=0、std=1）不同，是上游现状，如实锁定。
    rewards = torch.tensor([[5.0]])
    eos_mask = torch.ones(1, 1)
    adv, _ = ca.compute_grpo_per_token_advantage(rewards, eos_mask, [0])
    assert torch.allclose(adv, torch.tensor([[0.0]]), atol=1e-6)
