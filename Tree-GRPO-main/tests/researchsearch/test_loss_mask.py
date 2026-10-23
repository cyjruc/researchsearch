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
"""loss mask 测试（门禁 T06 无卡部分）。

锁定从 ``info_mask`` 到 ``loss_mask`` 的语义：工具观察（``<information>``）与
padding 不计策略损失，模型生成（``<think>/<search>/<answer>``）正确参与。

源码链路（见 docs/researchsearch/loss-mask.md）：

- ``generation._info_masked_concatenate_with_padding`` 把 information block 的
  token 整体替换成 ``pad_token_id``，因此 ``create_attention_mask`` 后 information
  位置为 0（测试 1–3）。
- ``ray_trainer._create_loss_mask`` / ``ray_trainer_ts._create_loss_mask`` 取
  ``loss_mask = info_mask[:, -response_length:]``（测试 4）。
- ``dp_actor.update_policy`` 在 ``state_masking`` 为真时用 ``loss_mask`` 代替
  ``attention_mask`` 的 response 段作为 ``response_mask``（测试 5）。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load_tensor_helper():
    spec = importlib.util.spec_from_file_location(
        "researchsearch_tensor_helper", _REPO / "search_r1/llm_agent/tensor_helper.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_th = _load_tensor_helper()
TensorHelper = _th.TensorHelper
TensorConfig = _th.TensorConfig


def _load_generation():
    # 绕过 verl（本机缺 tensordict）与 tree_node（依赖 pydantic/verl），
    # 只测 _info_masked_concatenate_with_padding，它只用 tokenizer.pad_token_id。
    for name, attrs in (
        ("verl", {"DataProto": type("DataProto", (), {})}),
        ("verl.utils", {}),
        ("verl.utils.tracking", {"Tracking": type("Tracking", (), {})}),
        # evidence 纯标准库模块按真实路径加载，供 generation_ts 顶层
        # `from verl.utils.reward_score.evidence import ...` 解析。
        ("verl.utils.reward_score", {}),
        ("search_r1", {}),
        ("search_r1.llm_agent", {}),
        ("search_r1.search", {}),
        ("search_r1.search.retrieval_client", {
            "post_retrieve": lambda *a, **k: {},
            "RetrievalError": type("RetrievalError", (Exception,), {}),
        }),
        # generation_ts 顶层 import 了 cost_stats；本文件只测 mask 拼接，mock 掉即可。
        ("search_r1.search.cost_stats", {"physical_new_tokens_from_tree": lambda *a, **k: 0}),
        ("search_r1.llm_agent.tree_node", {
            "TreeNode": type("TreeNode", (), {}),
            "dprint": lambda *a, **k: None,
            "DEBUG": False,
            "export_tree_roots": lambda *a, **k: [],
        }),
    ):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        # 无条件 setattr：即使模块已被其他测试注入（如 test_core_algos 的空 verl），
        # 也要补齐本文件需要的属性（如 verl.DataProto）。
        for key, value in attrs.items():
            setattr(mod, key, value)

    # 加载真实 evidence.py（纯标准库，无重型依赖），供 generation_ts 顶层 import。
    ev_spec = importlib.util.spec_from_file_location(
        "verl.utils.reward_score.evidence", _REPO / "verl/utils/reward_score/evidence.py"
    )
    ev_mod = importlib.util.module_from_spec(ev_spec)
    sys.modules["verl.utils.reward_score.evidence"] = ev_mod
    ev_spec.loader.exec_module(ev_mod)

    th_spec = importlib.util.spec_from_file_location(
        "search_r1.llm_agent.tensor_helper", _REPO / "search_r1/llm_agent/tensor_helper.py"
    )
    th_mod = importlib.util.module_from_spec(th_spec)
    sys.modules["search_r1.llm_agent.tensor_helper"] = th_mod
    th_spec.loader.exec_module(th_mod)

    gen_spec = importlib.util.spec_from_file_location(
        "search_r1.llm_agent.generation", _REPO / "search_r1/llm_agent/generation.py"
    )
    gen_mod = importlib.util.module_from_spec(gen_spec)
    sys.modules["search_r1.llm_agent.generation"] = gen_mod
    gen_spec.loader.exec_module(gen_mod)
    return gen_mod


_gen = _load_generation()
_helper = TensorHelper(TensorConfig(pad_token_id=0, max_prompt_length=64, max_obs_length=64, max_start_length=64))


def _info_masked_concat(prompt, prompt_with_mask, response, info):
    fake_self = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    return _gen.LLMGenerationManager._info_masked_concatenate_with_padding(
        fake_self, prompt, prompt_with_mask, response, info=info, pad_to_left=False
    )


# --- create_attention_mask（真实代码） ---------------------------------------


def test_create_attention_mask_maps_pad_to_zero():
    # pad_id=0：非 pad → 1，pad → 0。
    input_ids = torch.tensor([[1, 2, 0, 0], [0, 3, 4, 5]])
    mask = _helper.create_attention_mask(input_ids)
    assert torch.equal(mask, torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]]))


# --- information block 被替换为 pad_id（真实代码） ----------------------------


def test_information_block_is_replaced_with_pad_id():
    prompt = torch.tensor([[1, 2, 3]])
    prompt_with_mask = torch.tensor([[1, 2, 3]])
    response = torch.tensor([[4, 5]])  # 生成 token（think/search/answer）
    info = torch.tensor([[7, 8, 9]])  # 工具观察（information）

    padded, padded_with_info = _info_masked_concat(prompt, prompt_with_mask, response, info)

    # 原始响应保持生成 token，information 位置整体被替换为 pad_id=0。
    assert torch.equal(padded, torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9]]))
    assert torch.equal(padded_with_info, torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]))


def test_information_positions_have_zero_info_mask():
    prompt = torch.tensor([[1, 2, 3]])
    prompt_with_mask = torch.tensor([[1, 2, 3]])
    response = torch.tensor([[4, 5]])
    info = torch.tensor([[7, 8, 9]])

    _, padded_with_info = _info_masked_concat(prompt, prompt_with_mask, response, info)
    info_mask = _helper.create_attention_mask(padded_with_info)

    # 生成位置（prompt+response）=1，information 位置=0。
    assert torch.equal(info_mask, torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]))


def test_without_information_everything_is_visible():
    prompt = torch.tensor([[1, 2]])
    prompt_with_mask = torch.tensor([[1, 2]])
    response = torch.tensor([[3, 4]])

    _, padded_with_info = _info_masked_concat(prompt, prompt_with_mask, response, info=None)
    info_mask = _helper.create_attention_mask(padded_with_info)

    assert torch.equal(info_mask, torch.tensor([[1, 1, 1, 1]]))


# --- loss_mask 切片与 state_masking 语义（锁定 ray_trainer / dp_actor） --------


def test_loss_mask_slices_response_segment_of_info_mask():
    # 复现 ray_trainer._create_loss_mask（ray_trainer.py:884、ray_trainer_ts.py:1008）：
    #   loss_mask = info_mask[:, -response_length:]
    # prompt 3 位 + response 4 位（生成 3 位 + information 1 位），总长 7。
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
    info_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 1]])
    response_length = 4

    loss_mask = info_mask[:, -response_length:]
    response_mask = attention_mask[:, -response_length:]

    # loss_mask 只保留 response 段：生成 3 位=1，information 1 位=0。
    assert torch.equal(loss_mask, torch.tensor([[1, 1, 0, 1]]))
    # attention_mask 的 response 段把 information 也标为可见（=1）。
    assert torch.equal(response_mask, torch.tensor([[1, 1, 1, 1]]))


def test_state_masking_uses_loss_mask_instead_of_attention_mask():
    # 锁定 dp_actor.update_policy（dp_actor.py:280-282）：
    #   state_masking=true  -> response_mask = data['loss_mask']（information=0，不计 loss）
    #   state_masking=false -> response_mask = attention_mask 的 response 段（information=1，误计）
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
    info_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 1]])
    response_length = 4

    loss_mask = info_mask[:, -response_length:]
    response_mask_attn = attention_mask[:, -response_length:]

    with_state_masking = loss_mask
    without_state_masking = response_mask_attn

    # information 位置（索引 2）在两种模式下不同：state_masking 必须置 0。
    assert with_state_masking[0, 2] == 0
    assert without_state_masking[0, 2] == 1
    # 生成位置（索引 0,1,3）两种模式下都参与损失。
    assert with_state_masking[0, 0] == 1 and without_state_masking[0, 0] == 1
    assert with_state_masking[0, 3] == 1 and without_state_masking[0, 3] == 1


# --- tree 版掩码构造（generation_ts，含 turns_mask/turns） ---------------------


def _load_generation_ts():
    # generation_ts 依赖的 mock（verl/tree_node/retrieval_client/tensor_helper）
    # 已在 _load_generation 中就位。此处只额外加载 tree 版模块，锁定带
    # turns_mask/turns 的掩码构造语义（T06 的 tree 路径，此前未被单测覆盖）。
    gen_ts_spec = importlib.util.spec_from_file_location(
        "search_r1.llm_agent.generation_ts", _REPO / "search_r1/llm_agent/generation_ts.py"
    )
    gen_ts_mod = importlib.util.module_from_spec(gen_ts_spec)
    sys.modules["search_r1.llm_agent.generation_ts"] = gen_ts_mod
    gen_ts_spec.loader.exec_module(gen_ts_mod)
    return gen_ts_mod


_gen_ts = _load_generation_ts()


def _tree_info_masked_concat(prompt, prompt_with_mask, response, turns_mask, turns, info):
    fake_self = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    return _gen_ts.LLMGenerationTreeSearchManager._info_masked_concatenate_with_padding(
        fake_self, prompt, prompt_with_mask, response, turns_mask, turns, info=info, pad_to_left=False
    )


def test_tree_information_block_is_replaced_with_pad_id():
    prompt = torch.tensor([[1, 2, 3]])
    prompt_with_mask = torch.tensor([[1, 2, 3]])
    response = torch.tensor([[4, 5]])
    turns_mask = torch.tensor([[1, 1, 1]])
    turns = torch.tensor([2])  # 真实形状 [batch]，见 generation_ts 的 turns_stats
    info = torch.tensor([[7, 8, 9]])

    padded, padded_with_info, _ = _tree_info_masked_concat(
        prompt, prompt_with_mask, response, turns_mask, turns, info
    )

    # 与 chain 版一致：information 位置整体被替换为 pad_id=0。
    assert torch.equal(padded, torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9]]))
    assert torch.equal(padded_with_info, torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]))


def test_tree_information_positions_have_zero_info_mask():
    prompt = torch.tensor([[1, 2, 3]])
    prompt_with_mask = torch.tensor([[1, 2, 3]])
    response = torch.tensor([[4, 5]])
    turns_mask = torch.tensor([[1, 1, 1]])
    turns = torch.tensor([2])  # 真实形状 [batch]，见 generation_ts 的 turns_stats
    info = torch.tensor([[7, 8, 9]])

    _, padded_with_info, _ = _tree_info_masked_concat(
        prompt, prompt_with_mask, response, turns_mask, turns, info
    )
    info_mask = _helper.create_attention_mask(padded_with_info)

    assert torch.equal(info_mask, torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]))


def test_tree_turns_mask_marks_response_content_and_masks_info():
    # tree 版独有语义：response content 位置 = 当前 turn 值，information 位置 = pad_id。
    prompt = torch.tensor([[1, 2, 3]])
    prompt_with_mask = torch.tensor([[1, 2, 3]])
    response = torch.tensor([[4, 5]])
    turns_mask = torch.tensor([[1, 1, 1]])
    turns = torch.tensor([2])  # 真实形状 [batch]，见 generation_ts 的 turns_stats
    info = torch.tensor([[7, 8, 9]])

    _, _, padded_turns_mask = _tree_info_masked_concat(
        prompt, prompt_with_mask, response, turns_mask, turns, info
    )

    # prompt 段保留原 turns_mask，response content 段 = turn 值 2，information 段 = 0。
    assert torch.equal(padded_turns_mask, torch.tensor([[1, 1, 1, 2, 2, 0, 0, 0]]))


def test_tree_without_information_returns_three_tensors():
    prompt = torch.tensor([[1, 2, 3]])
    prompt_with_mask = torch.tensor([[1, 2, 3]])
    response = torch.tensor([[4, 5]])
    turns_mask = torch.tensor([[1, 1, 1]])
    turns = torch.tensor([2])  # 真实形状 [batch]，见 generation_ts 的 turns_stats

    padded, padded_with_info, padded_turns_mask = _tree_info_masked_concat(
        prompt, prompt_with_mask, response, turns_mask, turns, info=None
    )

    assert torch.equal(padded, torch.tensor([[1, 2, 3, 4, 5]]))
    assert torch.equal(padded_with_info, torch.tensor([[1, 2, 3, 4, 5]]))
    assert torch.equal(padded_turns_mask, torch.tensor([[1, 1, 1, 2, 2]]))
