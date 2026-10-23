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
"""树结构导出测试（方案 Day 5 任务 2）。

锁定 ``TreeNode.to_dict`` 与 ``export_tree_roots`` 两个纯导出接口的语义，全部
无需 GPU：

- ``to_dict`` 递归导出节点拓扑（tree_uid / node_uid / depth / 叶子 / 活跃 /
  分数 / 子节点数），是离线核对「分支在 agent-step 边界扩展」的载体；
- ``to_dict(include_tokens=True)`` 额外带上 1-D token/mask 张量，供核对
  「子节点共享父节点前缀」不变量；
- ``export_tree_roots`` 每棵树写一个 JSON（文件名用 tree_uid），幂等建目录。

本文件真实加载 ``tree_node.py``（mock 掉 pydantic / verl），与
``test_tree_evidence_wiring.py`` 相同。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load_tree_node():
    # tree_node.py 只 import pydantic.BaseModel 与 verl.DataProto（均未在
    # 导出逻辑里实际实例化），mock 后即可单文件加载。
    for name, attrs in (
        ("pydantic", {"BaseModel": type("BaseModel", (), {})}),
        ("verl", {"DataProto": type("DataProto", (), {})}),
    ):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for key, value in attrs.items():
            setattr(mod, key, value)

    spec = importlib.util.spec_from_file_location(
        "researchsearch_tree_node", _REPO / "search_r1/llm_agent/tree_node.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["researchsearch_tree_node"] = mod
    spec.loader.exec_module(mod)
    return mod


_tree = _load_tree_node()
TreeNode = _tree.TreeNode
export_tree_roots = _tree.export_tree_roots


def _make_node(tree_uid, node_uid, depth=0, is_root=False, is_leaf=False, **tensors):
    return TreeNode(
        tree_uid=tree_uid,
        node_uid=node_uid,
        is_root=is_root,
        is_leaf=is_leaf,
        depth=depth,
        reward_mode="base",
        **tensors,
    )


# --- to_dict 拓扑导出 --------------------------------------------------------


def test_to_dict_records_topology_recursively():
    root = _make_node("t0", "r0", depth=0, is_root=True)
    leaf_a = _make_node("t0", "a", depth=1, is_leaf=True)
    leaf_b = _make_node("t0", "b", depth=1, is_leaf=True)
    mid = _make_node("t0", "m", depth=1)
    grand = _make_node("t0", "g", depth=2, is_leaf=True)
    mid.add_child(grand)
    root.add_child(leaf_a)
    root.add_child(mid)
    root.add_child(leaf_b)

    d = root.to_dict()

    assert d["tree_uid"] == "t0"
    assert d["node_uid"] == "r0"
    assert d["is_root"] is True
    assert d["is_leaf"] is False
    assert d["n_children"] == 3
    assert [c["node_uid"] for c in d["children"]] == ["a", "m", "b"]

    # 中间节点 m 之下还有孙节点 g：递归导出。
    m = d["children"][1]
    assert m["depth"] == 1
    assert m["n_children"] == 1
    assert m["children"][0]["node_uid"] == "g"
    assert m["children"][0]["depth"] == 2
    assert m["children"][0]["is_leaf"] is True


def test_to_dict_records_scores_and_stats():
    root = _make_node("t0", "r0", is_root=True)
    leaf = _make_node(
        "t0", "a", is_leaf=True, valid_action_stats=3, valid_search_stats=1
    )
    leaf.original_score = 0.8
    leaf.final_score = 0.5
    root.add_child(leaf)

    d = root.to_dict()
    c = d["children"][0]

    assert c["valid_action_stats"] == 3
    assert c["valid_search_stats"] == 1
    assert c["original_score"] == 0.8
    assert c["final_score"] == 0.5


def test_to_dict_default_lengths_are_zero_for_none_tensors():
    node = _make_node("t0", "r0", is_root=True)  # 未给任何张量
    d = node.to_dict()
    assert d["input_ids_len"] == 0
    assert d["responses_len"] == 0


# --- include_tokens 与共享前缀核对 ------------------------------------------


def test_to_dict_include_tokens_roundtrips_tensors():
    node = _make_node(
        "t0", "r0", is_root=True,
        input_ids=torch.tensor([1, 2, 3]),
        responses=torch.tensor([10, 20]),
        responses_with_info_mask=torch.tensor([10, 0]),  # 第二个 token 是观测，被掩码
        turns_mask=torch.tensor([1, 0]),
    )
    d = node.to_dict(include_tokens=True)

    assert d["input_ids_len"] == 3
    assert d["responses_len"] == 2
    assert d["input_ids"] == [1, 2, 3]
    assert d["responses"] == [10, 20]
    assert d["responses_with_info_mask"] == [10, 0]
    assert d["turns_mask"] == [1, 0]


def test_to_dict_captures_shared_prefix_between_parent_and_child():
    # 树搜索不变量：子节点 input_ids / responses 以父节点为前缀（共享前缀复用）。
    parent = _make_node(
        "t0", "r0", is_root=True,
        input_ids=torch.tensor([1, 2, 3]),
        responses=torch.tensor([10, 20]),
        responses_with_info_mask=torch.tensor([10, 20]),
        turns_mask=torch.tensor([1, 1]),
    )
    child = _make_node(
        "t0", "c0", depth=1, is_leaf=True,
        input_ids=torch.tensor([1, 2, 3, 4, 5]),          # 父前缀 + 本步动作
        responses=torch.tensor([10, 20, 30, 40]),          # 父前缀 + 本步动作
        responses_with_info_mask=torch.tensor([10, 20, 30, 0]),  # 观测被掩码
        turns_mask=torch.tensor([1, 1, 2, 0]),
    )
    parent.add_child(child)

    d = child.to_dict(include_tokens=True)
    p = parent.to_dict(include_tokens=True)

    # 子节点 token 前缀 == 父节点 token 前缀。
    assert d["input_ids"][: p["input_ids_len"]] == p["input_ids"]
    assert d["responses"][: p["responses_len"]] == p["responses"]
    # 观测块在 responses_with_info_mask 里被掩码为 pad(0)。
    assert d["responses_with_info_mask"][-1] == 0


# --- export_tree_roots 写盘 --------------------------------------------------


def test_export_tree_roots_writes_one_json_per_tree(tmp_path):
    r1 = _make_node("tree-a", "r1", is_root=True)
    r1.add_child(_make_node("tree-a", "a1", is_leaf=True))
    r2 = _make_node("tree-b", "r2", is_root=True)
    r2.add_child(_make_node("tree-b", "b1", is_leaf=True))

    paths = export_tree_roots([r1, r2], str(tmp_path))

    assert len(paths) == 2
    files = {p.name for p in tmp_path.iterdir()}
    assert files == {"tree-a.json", "tree-b.json"}

    # 写出的 JSON 与 to_dict 一致，且带 token 字段。
    with open(tmp_path / "tree-a.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["tree_uid"] == "tree-a"
    assert data["n_children"] == 1
    assert data["children"][0]["node_uid"] == "a1"


def test_export_tree_roots_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "trees"
    paths = export_tree_roots([_make_node("t", "r", is_root=True)], str(target))
    assert (target / "t.json").exists()
    assert paths == [str(target / "t.json")]


# --- log_prob_list 可变默认参数修复 -----------------------------------------


def test_log_prob_list_is_not_shared_between_instances():
    # 构造时不传 log_prob_list：修复后每个实例应有独立 list，互不污染。
    n1 = _make_node("t0", "n1")
    n2 = _make_node("t0", "n2")
    n1.log_prob_list.append(0.5)

    assert n2.log_prob_list == []
    assert n1.log_prob_list == [0.5]
