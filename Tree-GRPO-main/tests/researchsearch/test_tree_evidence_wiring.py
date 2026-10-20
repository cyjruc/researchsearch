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
"""树内证据奖励接线预备测试（方案 Day 5 任务 6 / Day 6 任务 1）。

为 Day 6 树内证据接线建立**固定测试样例**，锁定三个不变量（对应阻塞 8 /
已知陷阱 #2），全部无需 GPU：

1. 叶子顺序对齐：``final_node_list`` 按 root 分组、每组 ``k`` 个叶子，与
   ``gen_batch.repeat(k, interleave=True)`` 的行序（``np.repeat``）一一对应，
   因此证据字段沿根样本 repeat 传播后，其行序与叶子顺序天然对齐、不串题。
2. doc_id 提取：Wiki 检索原始返回里的 ``document.id`` 需旁路保留（``_passages2string``
   目前丢弃它），由 ``evidence.extract_doc_ids_from_retrieval`` 提取。
3. 证据奖励不串题：两题金标准证据不同、分支检索结果不同，逐叶计算互不影响。

本文件真实加载 ``tree_node.py``（mock 掉 pydantic / verl），并复用已交付的
``evidence.py`` 纯函数，不动核心训练路径。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load_tree_node():
    # tree_node.py 只 import pydantic.BaseModel 与 verl.DataProto（均未在
    # TreeNode 逻辑里实际实例化），mock 后即可单文件加载。
    mocks = (
        ("pydantic", {"BaseModel": type("BaseModel", (), {})}),
        ("verl", {"DataProto": type("DataProto", (), {})}),
    )
    for name, attrs in mocks:
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


def _load_evidence():
    spec = importlib.util.spec_from_file_location(
        "researchsearch_evidence", _REPO / "verl/utils/reward_score/evidence.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tree = _load_tree_node()
TreeNode = _tree.TreeNode
_evidence = _load_evidence()
extract_doc_ids_from_retrieval = _evidence.extract_doc_ids_from_retrieval
compute_evidence_reward = _evidence.compute_evidence_reward


def _make_leaf(tree_uid, node_uid, depth=1):
    return TreeNode(
        tree_uid=tree_uid,
        node_uid=node_uid,
        is_root=False,
        depth=depth,
        is_leaf=True,
        reward_mode="base",
        log_prob_list=[],
    )


def _make_root(tree_uid, node_uid):
    return TreeNode(
        tree_uid=tree_uid,
        node_uid=node_uid,
        is_root=True,
        depth=0,
        reward_mode="base",
        log_prob_list=[],
    )


# --- 叶子顺序对齐（不变式 1） ------------------------------------------------


def test_sample_leaf_returns_k_leaves_in_bfs_order():
    # 候选叶子数 == k 时，sample_leaf 返回全部叶子，顺序为 get_subtree_nodes 的
    # BFS 顺序（与 random.shuffle 无关）。
    root = _make_root("t0", "r0")
    leaf_a = _make_leaf("t0", "a")
    leaf_b = _make_leaf("t0", "b")
    root.add_child(leaf_a)
    root.add_child(leaf_b)

    leaves = root.sample_leaf(2)

    assert [n.node_uid for n in leaves] == ["a", "b"]
    assert all(n.is_leaf for n in leaves)


def test_leaf_order_aligns_with_interleaved_repeat():
    # 核心不变量：final_node_list 按 root 分组（每组 k 个叶子），与
    # gen_batch.repeat(k, interleave=True) 的 np.repeat 行序一一对应。
    # 因此证据字段沿根样本 repeat 传播后，其行序与叶子顺序天然对齐、不串题。
    roots = [_make_root("t0", "r0"), _make_root("t1", "r1")]
    for root, prefix in zip(roots, ("x", "y")):
        root.add_child(_make_leaf(root.tree_uid, prefix + "a"))
        root.add_child(_make_leaf(root.tree_uid, prefix + "b"))

    k = 2
    final_node_list = []
    for root in roots:
        final_node_list.extend(root.sample_leaf(k))

    # 复刻 DataProto.repeat(interleave=True) 的 non_tensor_batch 语义（np.repeat）。
    sample_ids = np.array(["s0", "s1"], dtype=object)
    repeated_sample_ids = np.repeat(sample_ids, k, axis=0)

    assert len(final_node_list) == 2 * k
    for i, node in enumerate(final_node_list):
        # 叶子 i 所属 root（tree_uid t0/t1）与 repeat 后第 i 行的样本 id（s0/s1）
        # 对齐：前 k 个叶子属于 root0 对应 s0，后 k 个属于 root1 对应 s1。
        expected_root = roots[i // k].tree_uid
        expected_sample = sample_ids[i // k]
        assert node.tree_uid == expected_root
        assert repeated_sample_ids[i] == expected_sample


# --- doc_id 提取（不变式 2） -------------------------------------------------


def test_extract_doc_ids_preserves_order_and_skips_missing():
    result = [
        {"document": {"id": "wiki:1", "contents": "a\nb"}, "score": 0.9},
        {"document": {"contents": "no id"}, "score": 0.8},
        {"document": {"id": "wiki:2", "contents": "c\nd"}, "score": 0.7},
        "not a doc dict",
    ]
    assert extract_doc_ids_from_retrieval(result) == ["wiki:1", "wiki:2"]


def test_extract_doc_ids_empty_or_malformed():
    assert extract_doc_ids_from_retrieval([]) == []
    assert extract_doc_ids_from_retrieval(None) == []
    assert extract_doc_ids_from_retrieval({"result": []}) == []


# --- 证据奖励不串题（不变式 3） ---------------------------------------------


def test_evidence_reward_two_questions_do_not_cross_contaminate():
    # 题 A：gold 2 个，可见命中 1 个 -> 0.5；题 B：gold 2 个，可见命中 2 个 -> 1.0。
    res_a = compute_evidence_reward(
        gold_doc_ids=["wiki:1", "wiki:2"],
        visible_doc_ids=["wiki:1", "wiki:3", "wiki:1"],  # 重复 wiki:1 只记一次
    )
    res_b = compute_evidence_reward(
        gold_doc_ids=["wiki:10", "wiki:11"],
        visible_doc_ids=["wiki:10", "wiki:11"],
    )

    assert res_a["score"] == pytest.approx(0.5)
    assert res_a["num_covered"] == 1
    assert res_b["score"] == pytest.approx(1.0)
    assert res_b["num_covered"] == 2
    # 两题 gold 完全不同，结果互不影响。
    assert set(res_a["gold_doc_ids"]).isdisjoint(res_b["gold_doc_ids"])


def test_evidence_reward_end_to_end_from_retrieval_result():
    # 端到端：Wiki 原始返回 -> doc_id -> 可见集合 -> 证据覆盖率。
    gold = ["wiki:1", "wiki:2"]
    visible_result = [
        {"document": {"id": "wiki:1", "contents": "a\nb"}, "score": 0.9},
        {"document": {"id": "wiki:9", "contents": "x\ny"}, "score": 0.8},
    ]
    visible_ids = extract_doc_ids_from_retrieval(visible_result)

    # 被截断、未进入观察的文档单独记 retrieval_hit，不计入分子。
    hit_result = [{"document": {"id": "wiki:2", "contents": "c\nd"}, "score": 0.6}]
    hit_ids = extract_doc_ids_from_retrieval(hit_result)

    res = compute_evidence_reward(
        gold_doc_ids=gold,
        visible_doc_ids=visible_ids,
        retrieval_hit_doc_ids=hit_ids,
    )

    assert res["score"] == pytest.approx(0.5)  # 只覆盖 wiki:1
    assert res["num_covered"] == 1
    assert res["num_retrieval_hit"] == 1  # wiki:2 被截断，不计入分子但单独报告
    assert res["visible_doc_ids"] == ["wiki:1", "wiki:9"]
