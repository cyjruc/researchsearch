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
"""E0 逐题导出测试（方案 Day 3 任务 1/2）。

锁定 ``LLMGenerationManager.run_llm_loop`` 在 ``is_validation`` 时累积逐题轨迹，
以及 ``generation.export_e0_trajectories`` 的写盘语义：

- 轨迹记录每轮动作 / 查询 / 观察 / 有效性，并按 ``dones`` 精确分类结束原因
  （``answer`` 提前结束 vs ``max_turns`` 预算耗尽）；
- 训练路径（``is_validation=False``）不累积轨迹，零开销；
- 逐题导出把轨迹与题目元信息（index / data_source / gold_answers）合并，每条
  题一个 JSON，文件名用 index。

本机缺 ``tensordict`` 无法 ``import verl``；沿用 mock 模式（参考
``test_loss_mask.py``），真实加载 ``generation.py``，但 mock 掉 ``run_llm_loop``
调用的重型依赖。
"""

from __future__ import annotations

import importlib.util
import json
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


class _FakeDataProto:
    """run_llm_loop 里用到的 DataProto 最小替身（仅 from_dict）。"""

    @classmethod
    def from_dict(cls, d):
        return SimpleNamespace(batch=d, meta_info={})


def _load_generation():
    _ensure_package("search_r1", _REPO / "search_r1")
    _ensure_package("search_r1.llm_agent", _REPO / "search_r1/llm_agent")
    _ensure_package("search_r1.search", _REPO / "search_r1/search")

    mocks = (
        ("verl", {"DataProto": _FakeDataProto}),
        ("verl.utils", {}),
        ("verl.utils.tracking", {"Tracking": type("Tracking", (), {})}),
        ("search_r1.search.retrieval_client", {"post_retrieve": lambda *a, **k: None}),
        # generation.py 顶层 import 的三个名字；run_llm_loop 不实际使用 tree 节点。
        ("search_r1.llm_agent.tree_node", {
            "TreeNode": type("TreeNode", (), {}),
            "DEBUG": False,
            "dprint": lambda *a, **k: None,
        }),
    )
    for name, attrs in mocks:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for key, value in attrs.items():
            setattr(mod, key, value)

    # 真实加载 tensor_helper（无重型依赖）。
    spec = importlib.util.spec_from_file_location(
        "search_r1.llm_agent.tensor_helper", _REPO / "search_r1/llm_agent/tensor_helper.py"
    )
    th_mod = importlib.util.module_from_spec(spec)
    sys.modules["search_r1.llm_agent.tensor_helper"] = th_mod
    spec.loader.exec_module(th_mod)

    # 真实加载 generation.py（依赖已补齐）。
    spec = importlib.util.spec_from_file_location(
        "search_r1.llm_agent.generation", _REPO / "search_r1/llm_agent/generation.py"
    )
    gen_mod = importlib.util.module_from_spec(spec)
    sys.modules["search_r1.llm_agent.generation"] = gen_mod
    spec.loader.exec_module(gen_mod)
    return gen_mod


_gen = _load_generation()
LLMGenerationManager = _gen.LLMGenerationManager
export_e0_trajectories = _gen.export_e0_trajectories


def _make_manager(is_validation, execute_fn, responses_str):
    """用 __new__ 构造最小 manager，mock 掉 run_llm_loop 调用的每个依赖。"""
    mgr = LLMGenerationManager.__new__(LLMGenerationManager)
    mgr.is_validation = is_validation
    mgr.config = SimpleNamespace(
        max_turns=1, max_start_length=8, max_prompt_length=16, max_obs_length=8,
        num_gpus=1, no_think_rl=False, topk=3,
    )
    mgr.tokenizer = SimpleNamespace(pad_token="<pad>", pad_token_id=0, eos_token_id=2)
    mgr.tensor_fn = SimpleNamespace(
        cut_to_effective_len=lambda batch, keys=None: batch,
        _example_level_pad=lambda responses, responses_str, active_mask: (responses, responses_str),
    )
    mgr._generate_with_gpu_padding = lambda rollings_active: SimpleNamespace(
        batch={"responses": torch.tensor([[1]])}, meta_info={}
    )
    mgr._postprocess_responses = lambda responses: (torch.tensor([[1]]), responses_str)
    mgr.execute_predictions = execute_fn
    mgr._process_next_obs = lambda next_obs: torch.tensor([[0]])
    mgr._update_rolling_state = lambda rollings, responses_ids, next_obs_ids: rollings
    mgr._update_right_side = lambda right_side, responses_ids, next_obs_ids=None: right_side
    mgr._compose_final_output = lambda left, right, meta_info: SimpleNamespace(meta_info=meta_info)
    return mgr


def _run(mgr):
    gen_batch = SimpleNamespace(batch={
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
    })
    initial_input_ids = torch.tensor([[1, 2, 3]])
    return mgr.run_llm_loop(gen_batch=gen_batch, initial_input_ids=initial_input_ids)


# --- run_llm_loop：轨迹累积与结束原因分类 --------------------------------


def test_validation_records_answer_trajectory():
    # 模型在 step 0 直接给出 <answer>：done=1 -> stop_reason="answer"。
    def execute(predictions, pad_token, active_mask, do_search=True):
        return [""], [1], [1], [0]

    mgr = _make_manager(True, execute, ["<answer>Paris</answer>"])
    out = _run(mgr)

    traj = out.meta_info["trajectories"][0]
    assert traj["stop_reason"] == "answer"
    assert traj["final_answer"] == "Paris"
    assert len(traj["turns"]) == 1
    assert traj["turns"][0]["action"] == "answer"
    assert traj["turns"][0]["content"] == "Paris"
    assert traj["turns"][0]["done"] == 1
    # answer 结束的题不再参与后续轮，turns_stats 不计入。
    assert traj["tool_cost"]["turns"] == 0


def test_validation_records_max_turns_trajectory():
    # 模型每轮都只 search（done=0），跑满 max_turns=1 仍未 answer -> "max_turns"。
    def execute(predictions, pad_token, active_mask, do_search=True):
        return ["<information>doc</information>"], [0], [1], [1]

    mgr = _make_manager(True, execute, ["<search>query</search>"])
    out = _run(mgr)

    traj = out.meta_info["trajectories"][0]
    assert traj["stop_reason"] == "max_turns"
    assert traj["final_answer"] is None
    # 主循环 1 轮 + 末尾 do_search=False 一轮，共 2 条 turn。
    assert len(traj["turns"]) == 2
    assert all(t["action"] == "search" for t in traj["turns"])
    assert traj["turns"][0]["content"] == "query"
    assert traj["turns"][0]["observation"] == "<information>doc</information>"
    assert traj["turns"][0]["is_search"] == 1
    # 搜索次数只统计主循环（do_search=False 轮 is_search=1 但此处仍计入有效搜索）。
    assert traj["tool_cost"]["searches"] == 2
    assert traj["tool_cost"]["valid_actions"] == 2


def test_validation_records_two_searches_then_answer():
    # fixture：两轮有效搜索后正常 answer 结束 —— 精确对应 Day 3 验收
    # 「至少一条真实轨迹完成两轮搜索并正常结束」。真实模型失败时用该 fixture
    # 验证协议能正确记录，而不是用固定成功轨迹冒充真实 rollout。
    responses_seq = ["<search>q1</search>", "<search>q2</search>", "<answer>Paris</answer>"]
    idx = {"n": 0}

    def postprocess(responses):
        r = responses_seq[idx["n"]]
        idx["n"] += 1
        return torch.tensor([[1]]), [r]

    def execute(predictions, pad_token, active_mask, do_search=True):
        # 主循环（do_search=True）做有效搜索且未结束；final 轮给出答案结束。
        if do_search:
            return ["<information>doc</information>"], [0], [1], [1]
        return [""], [1], [1], [0]

    mgr = _make_manager(True, execute, responses_seq[0])
    mgr.config.max_turns = 2
    mgr._postprocess_responses = postprocess
    out = _run(mgr)

    traj = out.meta_info["trajectories"][0]
    assert traj["stop_reason"] == "answer"
    assert traj["final_answer"] == "Paris"
    assert [t["action"] for t in traj["turns"]] == ["search", "search", "answer"]
    assert [t["is_search"] for t in traj["turns"]] == [1, 1, 0]
    assert [t["done"] for t in traj["turns"]] == [0, 0, 1]
    # 两轮搜索；answer 结束的轮不计入 turns（continuing search turns）。
    assert traj["tool_cost"]["searches"] == 2
    assert traj["tool_cost"]["turns"] == 2
    assert traj["tool_cost"]["valid_actions"] == 3


def test_training_does_not_accumulate_trajectories():
    # 训练路径（is_validation=False）不累积轨迹：meta_info 里的 trajectories 为空。
    def execute(predictions, pad_token, active_mask, do_search=True):
        return [""], [1], [1], [0]

    mgr = _make_manager(False, execute, ["<answer>Paris</answer>"])
    out = _run(mgr)

    assert out.meta_info["trajectories"] == []


# --- export_e0_trajectories：逐题写盘 ------------------------------------


def _trajectory(stop_reason, final_answer):
    return {
        "turns": [
            {"step": 0, "action": "search", "content": "q", "observation": "doc",
             "valid": 1, "is_search": 1, "done": 0},
            {"step": 1, "action": "answer", "content": final_answer, "observation": "",
             "valid": 1, "is_search": 0, "done": 1},
        ],
        "stop_reason": stop_reason,
        "final_answer": final_answer,
        "tool_cost": {"turns": 1, "searches": 1, "valid_actions": 2},
    }


def test_export_writes_one_json_per_question(tmp_path):
    trajectories = [
        _trajectory("answer", "Paris"),
        _trajectory("max_turns", None),
    ]
    non_tensor = {
        "index": [101, 102],
        "data_source": ["nq", "hotpotqa"],
        "reward_model": [
            {"ground_truth": {"target": ["Paris"]}, "style": "rule"},
            {"ground_truth": {"target": ["Beijing"]}, "style": "rule"},
        ],
    }

    paths = export_e0_trajectories(trajectories, non_tensor, str(tmp_path))

    assert len(paths) == 2
    assert {p.rsplit("/", 1)[-1] for p in paths} == {"e0_101.json", "e0_102.json"}

    with open(tmp_path / "e0_101.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["index"] == 101
    assert data["data_source"] == "nq"
    assert data["gold_answers"] == ["Paris"]
    assert data["stop_reason"] == "answer"
    assert data["final_answer"] == "Paris"
    assert data["tool_cost"]["searches"] == 1
    assert len(data["turns"]) == 2


def test_export_empty_trajectories_is_noop(tmp_path):
    paths = export_e0_trajectories([], {}, str(tmp_path))
    assert paths == []
    assert list(tmp_path.iterdir()) == []


def test_export_missing_metadata_is_safe(tmp_path):
    # gold 缺失 / index 缺失时回退，不抛异常。
    trajectories = [_trajectory("answer", "Paris")]
    non_tensor = {"index": [7], "data_source": ["nq"], "reward_model": [{}]}

    paths = export_e0_trajectories(trajectories, non_tensor, str(tmp_path))

    assert len(paths) == 1
    with open(tmp_path / "e0_7.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["gold_answers"] is None
    assert data["data_source"] == "nq"


def test_export_serializes_numpy_gold(tmp_path):
    # parquet 读出的 ground_truth.target 常是 numpy ndarray，必须能被 JSON 序列化。
    import numpy as np

    trajectories = [_trajectory("answer", "Paris")]
    non_tensor = {
        "index": np.array([7]),
        "data_source": np.array(["nq"]),
        "reward_model": np.array(
            [{"ground_truth": {"target": np.array(["Paris"])}}], dtype=object
        ),
    }

    paths = export_e0_trajectories(trajectories, non_tensor, str(tmp_path))

    assert len(paths) == 1
    with open(tmp_path / "e0_7.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["gold_answers"] == ["Paris"]
    assert data["index"] == 7
    assert data["data_source"] == "nq"

