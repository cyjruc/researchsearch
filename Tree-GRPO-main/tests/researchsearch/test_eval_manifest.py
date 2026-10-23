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
"""Multi-Hop Avg EM 测试集 manifest 预注册测试（架构 §2 / 方案 §10 第 9 项）。

锁定：

- 四项数据集缺一项 build 与 validate 都报错，不静默沿用前三项。
- 题目 ID 全局唯一（跨数据集重复报错）。
- 哈希覆盖除 hash 外全部内容，篡改内容使校验失败。
- 评分器注册与答案别名随 manifest 一起预注册。
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


manifest_mod = _load("verl/utils/reward_score/eval_manifest.py", "researchsearch_eval_manifest")


def _entry(qid, gold, source_id=None):
    e = {"question_id": qid, "golden_answers": gold}
    if source_id is not None:
        e["source_id"] = source_id
    return e


def _four_datasets():
    return {
        "hotpotqa": [_entry("h:1", ["Paris"], "h1")],
        "2wiki": [_entry("w:1", ["Tokyo"], "w1")],
        "musique": [_entry("m:1", ["x"], "m1")],
        "bamboogle": [_entry("b:1", ["y"], "b1")],
    }


# --- 构建与四项等权 -------------------------------------------------------------


def test_build_missing_dataset_raises():
    datasets = {"hotpotqa": [_entry("h:1", "a")]}
    with pytest.raises(ValueError, match="缺数据集"):
        manifest_mod.build_eval_manifest(datasets)


def test_build_produces_hash_and_scorer_registration():
    manifest = manifest_mod.build_eval_manifest(_four_datasets(), split="test")
    assert manifest["hash"].startswith("sha256:")
    assert manifest["scorer"]["module"] == "eval_metrics"
    assert manifest["scorer"]["em"] == "eval_metrics.em_match"
    assert manifest["datasets"]["hotpotqa"]["count"] == 1


def test_build_gold_string_becomes_alias_list():
    manifest = manifest_mod.build_eval_manifest(_four_datasets())
    entry = manifest["datasets"]["hotpotqa"]["entries"][0]
    assert entry["golden_answers"] == ["Paris"]


# --- 题目 ID 全局唯一 -----------------------------------------------------------


def test_build_duplicate_question_id_across_datasets_raises():
    datasets = _four_datasets()
    # 把 2wiki 的题目 ID 改成与 hotpotqa 相同。
    datasets["2wiki"] = [_entry("h:1", ["Tokyo"])]
    with pytest.raises(ValueError, match="跨数据集重复"):
        manifest_mod.build_eval_manifest(datasets)


# --- 哈希可复核 ------------------------------------------------------------------


def test_validate_roundtrip_passes():
    manifest = manifest_mod.build_eval_manifest(_four_datasets())
    manifest_mod.validate_eval_manifest(manifest)  # 不抛即通过


def test_validate_detects_tampered_content():
    manifest = manifest_mod.build_eval_manifest(_four_datasets())
    manifest["datasets"]["hotpotqa"]["entries"][0]["golden_answers"] = ["Berlin"]
    with pytest.raises(ValueError, match="hash 不匹配"):
        manifest_mod.validate_eval_manifest(manifest)


def test_validate_requires_hash():
    manifest = manifest_mod.build_eval_manifest(_four_datasets())
    del manifest["hash"]
    with pytest.raises(ValueError, match="hash"):
        manifest_mod.validate_eval_manifest(manifest)


def test_validate_missing_dataset_raises():
    manifest = manifest_mod.build_eval_manifest(_four_datasets())
    del manifest["datasets"]["bamboogle"]
    with pytest.raises(ValueError, match="缺数据集"):
        manifest_mod.validate_eval_manifest(manifest)


# --- data_version 统一 / 逐项 -----------------------------------------------------


def test_data_version_unified_string():
    manifest = manifest_mod.build_eval_manifest(_four_datasets(), data_version="wiki-2026-01")
    for dataset in manifest_mod.MULTI_HOP_DATASETS:
        assert manifest["datasets"][dataset]["data_version"] == "wiki-2026-01"


def test_data_version_per_dataset():
    versions = {"hotpotqa": "v1", "2wiki": "v2"}
    manifest = manifest_mod.build_eval_manifest(_four_datasets(), data_version=versions)
    assert manifest["datasets"]["hotpotqa"]["data_version"] == "v1"
    assert manifest["datasets"]["2wiki"]["data_version"] == "v2"
    assert manifest["datasets"]["musique"]["data_version"] == ""


def test_manifest_summary_counts():
    manifest = manifest_mod.build_eval_manifest(_four_datasets())
    assert manifest_mod.manifest_summary(manifest) == {
        "hotpotqa": 1,
        "2wiki": 1,
        "musique": 1,
        "bamboogle": 1,
    }
