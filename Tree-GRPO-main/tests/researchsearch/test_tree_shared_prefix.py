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
"""树搜索共享前缀对齐测试（方案 Day 5 任务 4）。

锁定 ``generation_ts`` 里共享前缀复用的核心不变量（全部无需 GPU）：

- ``_info_masked_concatenate_with_padding``：拼接「父前缀 + 本步动作 + 观测」
  时，父前缀 token 顺序不变地保留在最前；``<information>`` 观测块在
  ``responses_with_info_mask`` / ``turns_mask`` 中被掩码为 pad；
- ``_update_right_side``：子节点 right_side 是父节点 right_side 的超集
  （共享前缀复用），动作 token 打上正确的 turn 编号。

本机缺 ``tensordict`` 无法 ``import verl``；本文件 mock 掉 ``verl`` /
``verl.utils.tracking`` / ``search_r1.search.retrieval_client``，真实加载
``tree_node.py`` + ``tensor_helper.py`` + ``generation_ts.py``。
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


def _ensure_package(name, path):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod


def _load_generation_ts():
    _ensure_package("search_r1", _REPO / "search_r1")
    _ensure_package("search_r1.llm_agent", _REPO / "search_r1/llm_agent")
    _ensure_package("search_r1.search", _REPO / "search_r1/search")

    # mock 掉重型依赖（generation_ts 顶层 import 用到的全部名字）。
    mocks = (
        ("pydantic", {"BaseModel": type("BaseModel", (), {})}),
        ("verl", {"DataProto": type("DataProto", (), {})}),
        ("verl.utils", {}),
        ("verl.utils.tracking", {"Tracking": type("Tracking", (), {})}),
        # evidence 纯标准库模块按真实路径加载，供 generation_ts 顶层
        # `from verl.utils.reward_score.evidence import ...` 解析。
        ("verl.utils.reward_score", {}),
        ("search_r1.search.retrieval_client", {"post_retrieve": lambda *a, **k: None}),
        # generation_ts 顶层 import 了 cost_stats（成本接线落地）；本测试只锁共享前缀，
        # 不锁 token 计数口径，故 mock 掉 physical_new_tokens_from_tree。
        ("search_r1.search.cost_stats", {"physical_new_tokens_from_tree": lambda *a, **k: 0}),
    )
    for name, attrs in mocks:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for key, value in attrs.items():
            setattr(mod, key, value)

    # 加载真实 evidence.py（纯标准库，无重型依赖），供 generation_ts 顶层 import。
    ev_spec = importlib.util.spec_from_file_location(
        "verl.utils.reward_score.evidence", _REPO / "verl/utils/reward_score/evidence.py"
    )
    ev_mod = importlib.util.module_from_spec(ev_spec)
    sys.modules["verl.utils.reward_score.evidence"] = ev_mod
    ev_spec.loader.exec_module(ev_mod)

    # 真实加载 tree_node.py（依赖已 mock）。
    spec = importlib.util.spec_from_file_location(
        "search_r1.llm_agent.tree_node", _REPO / "search_r1/llm_agent/tree_node.py"
    )
    tree_mod = importlib.util.module_from_spec(spec)
    sys.modules["search_r1.llm_agent.tree_node"] = tree_mod
    spec.loader.exec_module(tree_mod)

    # 真实加载 tensor_helper.py（无重型依赖）。
    spec = importlib.util.spec_from_file_location(
        "search_r1.llm_agent.tensor_helper", _REPO / "search_r1/llm_agent/tensor_helper.py"
    )
    th_mod = importlib.util.module_from_spec(spec)
    sys.modules["search_r1.llm_agent.tensor_helper"] = th_mod
    spec.loader.exec_module(th_mod)

    # 真实加载 generation_ts.py（依赖已补齐）。
    spec = importlib.util.spec_from_file_location(
        "search_r1.llm_agent.generation_ts", _REPO / "search_r1/llm_agent/generation_ts.py"
    )
    gen_mod = importlib.util.module_from_spec(spec)
    sys.modules["search_r1.llm_agent.generation_ts"] = gen_mod
    spec.loader.exec_module(gen_mod)
    return gen_mod


_gen = _load_generation_ts()
LLMGenerationTreeSearchManager = _gen.LLMGenerationTreeSearchManager
TensorHelper = sys.modules["search_r1.llm_agent.tensor_helper"].TensorHelper
TensorConfig = sys.modules["search_r1.llm_agent.tensor_helper"].TensorConfig


def _make_manager():
    """绕过 __init__ 构造最小 manager，只保留被测方法用到的属性。"""
    mgr = LLMGenerationTreeSearchManager.__new__(LLMGenerationTreeSearchManager)
    mgr.tokenizer = SimpleNamespace(pad_token_id=0)
    mgr.config = SimpleNamespace(max_prompt_length=100)
    mgr.tensor_fn = TensorHelper(
        TensorConfig(pad_token_id=0, max_prompt_length=100, max_obs_length=100, max_start_length=100)
    )
    return mgr


# --- _info_masked_concatenate_with_padding ----------------------------------


def test_info_masked_concat_preserves_prefix_and_masks_info():
    mgr = _make_manager()
    # 父前缀 [11,12]（累积 responses）+ 本步动作 [21,22,23] + 观测 [99,98]。
    prompt = torch.tensor([[11, 12]])
    prompt_with_mask = torch.tensor([[11, 12]])
    response = torch.tensor([[21, 22, 23]])
    turns_mask = torch.tensor([[1, 1]])
    turns = torch.tensor([2])
    info = torch.tensor([[99, 98]])

    out, out_info, out_turns = mgr._info_masked_concatenate_with_padding(
        prompt, prompt_with_mask, response, turns_mask, turns, info, pad_to_left=False
    )

    # 共享前缀保留在最前，本步动作随后，观测最后。
    assert out.tolist() == [[11, 12, 21, 22, 23, 99, 98]]
    # 观测块在 responses_with_info_mask 里被掩码为 pad(0)。
    assert out_info.tolist() == [[11, 12, 21, 22, 23, 0, 0]]
    # turns_mask：父前缀保持 1，本步动作打上 turn=2，观测掩码为 0。
    assert out_turns.tolist() == [[1, 1, 2, 2, 2, 0, 0]]


def test_info_masked_concat_without_info_keeps_action_visible():
    mgr = _make_manager()
    prompt = torch.tensor([[11]])
    prompt_with_mask = torch.tensor([[11]])
    response = torch.tensor([[21, 22]])
    turns_mask = torch.tensor([[1]])
    turns = torch.tensor([2])

    out, out_info, out_turns = mgr._info_masked_concatenate_with_padding(
        prompt, prompt_with_mask, response, turns_mask, turns, info=None, pad_to_left=False
    )

    # 无观测：responses 与 responses_with_info_mask 完全一致（动作全部可见）。
    assert out.tolist() == [[11, 21, 22]]
    assert out_info.tolist() == [[11, 21, 22]]
    assert out_turns.tolist() == [[1, 2, 2]]


# --- _update_right_side ------------------------------------------------------


def test_update_right_side_extends_parent_prefix():
    mgr = _make_manager()
    right_side = {
        "responses": torch.tensor([[11, 12]]),
        "responses_with_info_mask": torch.tensor([[11, 12]]),
        "turns_mask": torch.tensor([[1, 1]]),
    }
    cur_responses = torch.tensor([[21, 22]])
    turns = torch.tensor([2])
    next_obs_ids = torch.tensor([[99]])

    out = mgr._update_right_side(right_side, cur_responses, turns, next_obs_ids)

    # 子节点 responses = 父前缀 [11,12] + 本步动作 [21,22] + 观测 [99]。
    assert out["responses"].tolist() == [[11, 12, 21, 22, 99]]
    # 观测被掩码。
    assert out["responses_with_info_mask"].tolist() == [[11, 12, 21, 22, 0]]
    # turns_mask：本步动作打 turn=2，观测 0。
    assert out["turns_mask"].tolist() == [[1, 1, 2, 2, 0]]


def test_update_right_side_truncates_to_max_prompt_length():
    mgr = _make_manager()
    mgr.config = SimpleNamespace(max_prompt_length=4)  # 收紧窗口以触发截断
    right_side = {
        "responses": torch.tensor([[11, 12]]),
        "responses_with_info_mask": torch.tensor([[11, 12]]),
        "turns_mask": torch.tensor([[1, 1]]),
    }
    cur_responses = torch.tensor([[21, 22, 23]])
    turns = torch.tensor([2])
    next_obs_ids = torch.tensor([[99]])

    out = mgr._update_right_side(right_side, cur_responses, turns, next_obs_ids)

    # right_truncate=True：保留最前 max_prompt_length=4 个 token。
    assert out["responses"].tolist() == [[11, 12, 21, 22]]
    assert out["responses_with_info_mask"].tolist() == [[11, 12, 21, 22]]
    assert out["turns_mask"].tolist() == [[1, 1, 2, 2]]
