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
"""优势计算路由测试（门禁 T07 / 方案 Day 5 任务 3）。

``test_core_algos.py`` 已锁定 ``core_algos`` 三个纯函数的数值语义；本文件补
**路由层**：``ray_trainer_ts.compute_advantage`` 在 ``adv_estimator`` 取值下如何
把 ``uid`` / ``tree_uid`` 分派给哪个 core_algos 函数。

核心事实（对齐方案 §6 Day 5 任务 3，防止误用非默认路由）：

- ``adv_estimator='tree'``（默认）＝两次 ``compute_grpo_outcome_advantage``
  （按题目 ``uid`` 一次、按树 ``tree_uid`` 一次）**相加**；
- ``adv_estimator='tree_2norm'``＝一次 ``compute_tree_grpo_outcome_advantage``
  （顺序双重标准化），二者数值不同，不能混用。

本机缺 ``tensordict`` 导致无法 ``import verl``；本文件 mock 掉 ``ray_trainer_ts``
顶部除 ``torch`` / ``core_algos`` 之外的 import，单文件加载后调用真实的
``compute_advantage``。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load_core_algos():
    # 与 test_core_algos.py 相同：core_algos 三个被测函数不使用 verl_F。
    for name in ("verl", "verl.utils"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    if "verl.utils.torch_functional" not in sys.modules:
        sys.modules["verl.utils.torch_functional"] = types.ModuleType("verl.utils.torch_functional")
    spec = importlib.util.spec_from_file_location(
        "researchsearch_core_algos_for_routing", _REPO / "verl/trainer/ppo/core_algos.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ray_trainer_ts():
    core_algos = _load_core_algos()

    # 无条件补齐 mock 属性：其他测试（如 test_core_algos）可能已注入空 verl 包，
    # 这里必须把 ray_trainer_ts 顶层 import 用到的名字全部补齐。
    mocks = (
        ("codetiming", {"Timer": type("Timer", (), {})}),
        ("omegaconf", {"OmegaConf": type("OmegaConf", (), {}), "open_dict": type("open_dict", (), {})}),
        ("verl", {"DataProto": type("DataProto", (), {})}),
        ("verl.protocol", {
            "pad_dataproto_to_divisor": lambda *a, **k: None,
            "unpad_dataproto": lambda *a, **k: None,
        }),
        ("verl.single_controller", {}),
        ("verl.single_controller.base", {"Worker": type("Worker", (), {})}),
        ("verl.single_controller.ray", {
            "RayResourcePool": type("RayResourcePool", (), {}),
            "RayWorkerGroup": type("RayWorkerGroup", (), {}),
            "RayClassWithInitArgs": type("RayClassWithInitArgs", (), {}),
        }),
        ("verl.single_controller.ray.base", {
            "create_colocated_worker_cls": lambda *a, **k: None,
        }),
        ("verl.trainer", {}),
        ("verl.trainer.ppo", {}),
        ("verl.utils.seqlen_balancing", {
            "get_seqlen_balanced_partitions": lambda *a, **k: [],
            "log_seqlen_unbalance": lambda *a, **k: None,
        }),
        ("verl.utils.torch_functional", {"masked_mean": lambda *a, **k: torch.tensor(0.0)}),
        ("search_r1", {}),
        ("search_r1.llm_agent", {}),
        ("search_r1.llm_agent.generation_ts", {
            "LLMGenerationTreeSearchManager": type("LLMGenerationTreeSearchManager", (), {}),
            "GenerationTreeSearchConfig": type("GenerationTreeSearchConfig", (), {}),
        }),
        ("search_r1.llm_agent.generation", {
            "LLMGenerationManager": type("LLMGenerationManager", (), {}),
            "GenerationConfig": type("GenerationConfig", (), {}),
        }),
        ("search_r1.search", {}),
        ("search_r1.search.cost_stats", {
            "CostStats": type("CostStats", (), {}),
            "directory_size_bytes": lambda *a, **k: 0,
        }),
    )
    for name, attrs in mocks:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for key, value in attrs.items():
            setattr(mod, key, value)

    # 真实 core_algos 挂到 ray_trainer_ts 会 import 的位置。
    sys.modules["verl.trainer.ppo.core_algos"] = core_algos

    spec = importlib.util.spec_from_file_location(
        "researchsearch_ray_trainer_ts", _REPO / "verl/trainer/ppo/ray_trainer_ts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["researchsearch_ray_trainer_ts"] = mod
    spec.loader.exec_module(mod)
    return mod


_trainer = _load_ray_trainer_ts()
_compute_advantage = _trainer.compute_advantage
_core_algos = _trainer.core_algos


def _make_data(rewards, uid, tree_uid):
    """构造 compute_advantage 只需访问字段的假 DataProto。"""
    responses = torch.zeros(rewards.shape, dtype=torch.long)
    attention_mask = torch.ones(rewards.shape)
    return SimpleNamespace(
        batch={
            "token_level_rewards": rewards,
            "responses": responses,
            "attention_mask": attention_mask,
        },
        non_tensor_batch={"uid": uid, "tree_uid": tree_uid},
    )


def test_tree_routing_adds_two_grpo_normalizations():
    # 默认 tree = inter(按 uid) + inner(按 tree_uid)，两个中间值也写回 batch。
    rewards = torch.tensor([[1.0], [3.0], [10.0], [30.0]])
    uid = [0, 0, 1, 1]
    tree_uid = [0, 1, 1, 0]
    data = _make_data(rewards, uid, tree_uid)

    _compute_advantage(data, "tree")

    response_mask = torch.ones_like(rewards)
    inter, _ = _core_algos.compute_grpo_outcome_advantage(rewards, response_mask, uid)
    inner, _ = _core_algos.compute_grpo_outcome_advantage(rewards, response_mask, tree_uid)

    assert torch.allclose(data.batch["inter_tree_advantages"], inter)
    assert torch.allclose(data.batch["inner_tree_advantages"], inner)
    assert torch.allclose(data.batch["advantages"], inter + inner)
    assert torch.allclose(data.batch["returns"], inter + inner)


def test_tree_2norm_routing_uses_compute_tree_grpo():
    # tree_2norm 是一次顺序双重标准化，不是两次独立标准化相加。
    rewards = torch.tensor([[1.0], [3.0], [10.0], [30.0]])
    uid = [0, 0, 1, 1]
    tree_uid = [0, 1, 1, 0]
    data = _make_data(rewards, uid, tree_uid)

    _compute_advantage(data, "tree_2norm")

    response_mask = torch.ones_like(rewards)
    tree2, _ = _core_algos.compute_tree_grpo_outcome_advantage(rewards, response_mask, uid, tree_uid)

    assert torch.allclose(data.batch["advantages"], tree2)
    assert torch.allclose(data.batch["returns"], tree2)
    # tree_2norm 不产生 tree 路由特有的中间字段。
    assert "inter_tree_advantages" not in data.batch
    assert "inner_tree_advantages" not in data.batch


def test_tree_routing_differs_from_tree_2norm():
    # 同一组数值下，tree 与 tree_2norm 结果不同，锁定"不能误当默认算法"。
    rewards = torch.tensor([[1.0], [3.0], [10.0], [30.0]])
    uid = [0, 0, 1, 1]
    tree_uid = [0, 1, 1, 0]

    tree_data = _make_data(rewards, uid, tree_uid)
    _compute_advantage(tree_data, "tree")

    tree2_data = _make_data(rewards, uid, tree_uid)
    _compute_advantage(tree2_data, "tree_2norm")

    assert not torch.allclose(tree_data.batch["advantages"], tree2_data.batch["advantages"], atol=1e-3)


def test_grpo_routing_single_normalization():
    # 普通 GRPO 路由按 uid 标准化一次，与 tree 路由对照确认分派正确。
    rewards = torch.tensor([[1.0], [3.0], [10.0], [30.0]])
    uid = [0, 0, 1, 1]
    data = _make_data(rewards, uid, [0, 1, 1, 0])  # tree_uid 不被 grpo 使用

    _compute_advantage(data, "grpo")

    response_mask = torch.ones_like(rewards)
    grpo, _ = _core_algos.compute_grpo_outcome_advantage(rewards, response_mask, uid)
    assert torch.allclose(data.batch["advantages"], grpo)
    assert torch.allclose(data.batch["returns"], grpo)
