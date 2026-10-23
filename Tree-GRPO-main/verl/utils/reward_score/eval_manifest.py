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
"""Multi-Hop Avg EM 测试集 manifest 预注册（架构 §2 / 第 2 周方案 §10 第 9 项）。

架构 §2 要求：正式评测前把每项数据版本、split、题目原始 ID、筛选规则、答案别名及
归一化评分代码写入只读评测 manifest，并记录哈希；E0–E3 的 Wiki checkpoint 使用
完全相同的题目 ID 与评分器。

本模块零 verl 依赖（只用标准库），便于无 GPU / 无语料阶段交付「schema + 构建 +
校验 + 哈希」，真实题目 ID 与数据版本待测试集文件核对后回填（生成脚本
``scripts/researchsearch/gen_eval_manifest.py`` 只填骨架，不伪称已预注册）。

锁定的口径：

- Multi-Hop Avg EM 固定四项（hotpotqa / 2wiki / musique / bamboogle），缺一项
  build 与 validate 都报错，不静默沿用前三项（架构 §2）。
- 每题 ``question_id`` 全局唯一，跨数据集不重复。
- 哈希覆盖除 ``hash`` 外的全部内容，篡改题目/别名/评分器/版本均使校验失败。
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Optional, Sequence, Union

MULTI_HOP_DATASETS = ("hotpotqa", "2wiki", "musique", "bamboogle")

_DEFAULT_NORMALIZE_NOTE = "去冠词/标点/小写/空白，与 qa_em / qa_f1_format 一致"


def _gold_list(golden_answers) -> list[str]:
    if isinstance(golden_answers, str):
        return [golden_answers]
    if golden_answers is None:
        return []
    return [str(g) for g in golden_answers]


def _data_version_for(data_version, dataset: str) -> str:
    if isinstance(data_version, str):
        return data_version
    if isinstance(data_version, Mapping):
        return data_version.get(dataset, "")
    return ""


def _build_entries(dataset: str, entries: Sequence[dict]) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            raise ValueError(f"{dataset} 的条目不是对象：{e!r}")
        if "question_id" not in e or not isinstance(e["question_id"], str) or not e["question_id"]:
            raise ValueError(f"{dataset} 有条目缺少非空 question_id")
        gold = _gold_list(e.get("golden_answers"))
        if not gold:
            raise ValueError(f"{dataset} 条目 question_id={e['question_id']!r} 缺少 golden_answers")
        out.append(
            {
                "question_id": e["question_id"],
                "golden_answers": gold,
                "source_id": e.get("source_id"),
            }
        )
    return out


def build_eval_manifest(
    datasets: Mapping[str, Sequence[dict]],
    *,
    split: str = "test",
    data_version: Union[str, Mapping[str, str], None] = None,
    filter_rules: str = "",
    scorer_module: str = "eval_metrics",
    scorer_version: str = "1",
    normalize_note: str = _DEFAULT_NORMALIZE_NOTE,
) -> dict:
    """构建只读评测 manifest 并写入哈希。

    Args:
        datasets: ``{dataset: [{"question_id", "golden_answers", "source_id"?}, ...]}``，
            键必须是 Multi-Hop Avg EM 的四项数据集（缺项报错）。
        split: 每题来自的官方 split（默认 test，正式评测锁定）。
        data_version: 统一数据版本字符串，或 ``{dataset: version}`` 逐项指定。
        filter_rules: 筛选规则说明（写进 manifest 供复核）。
        scorer_module / scorer_version / normalize_note: 评分器注册信息。

    Returns:
        manifest dict（含 ``hash``）；``datasets`` 缺项抛 ``ValueError``。
    """
    missing = [d for d in MULTI_HOP_DATASETS if d not in datasets]
    if missing:
        raise ValueError(
            f"缺数据集：{missing}（Multi-Hop Avg EM 四项等权，缺一项不能预注册）"
        )

    manifest_datasets: dict = {}
    seen_ids: set[str] = set()
    for dataset in MULTI_HOP_DATASETS:
        entries = _build_entries(dataset, datasets[dataset])
        for e in entries:
            qid = e["question_id"]
            if qid in seen_ids:
                raise ValueError(f"question_id 跨数据集重复：{qid!r}")
            seen_ids.add(qid)
        manifest_datasets[dataset] = {
            "data_version": _data_version_for(data_version, dataset),
            "split": split,
            "filter_rules": filter_rules,
            "count": len(entries),
            "entries": entries,
        }

    manifest = {
        "schema_version": 1,
        "scorer": {
            "module": scorer_module,
            "version": scorer_version,
            "normalize": normalize_note,
            "em": "eval_metrics.em_match",
            "f1": "eval_metrics.f1_match",
        },
        "datasets": manifest_datasets,
    }
    manifest["hash"] = compute_manifest_hash(manifest)
    return manifest


def compute_manifest_hash(manifest: Mapping) -> str:
    """对除 ``hash`` 外的确定性内容算 SHA-256，返回 ``sha256:<hex>``。"""
    payload = {k: v for k, v in manifest.items() if k != "hash"}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def validate_eval_manifest(
    manifest: Mapping,
    *,
    multi_hop_datasets: Sequence[str] = MULTI_HOP_DATASETS,
) -> None:
    """校验 manifest 合法；非法抛 ``ValueError``，合法返回 None。

    校验：四项数据集齐、题目 ID 全局唯一、每题 golden_answers 非空、评分器已注册、
    哈希与内容一致。
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest 不是对象")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("manifest 缺少 datasets")
    missing = [d for d in multi_hop_datasets if d not in datasets]
    if missing:
        raise ValueError(f"manifest 缺数据集：{missing}")

    if not isinstance(manifest.get("scorer"), Mapping) or not manifest["scorer"].get("module"):
        raise ValueError("manifest 缺少 scorer 注册")

    if "hash" not in manifest:
        raise ValueError("manifest 缺少 hash")

    seen_ids: set[str] = set()
    for dataset in multi_hop_datasets:
        entries = datasets[dataset].get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"{dataset} 缺少 entries 列表")
        for e in entries:
            qid = e.get("question_id")
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"{dataset} 有条目缺少 question_id")
            if qid in seen_ids:
                raise ValueError(f"question_id 跨数据集重复：{qid!r}")
            seen_ids.add(qid)
            if not e.get("golden_answers"):
                raise ValueError(f"question_id={qid!r} 缺少 golden_answers")

    if manifest["hash"] != compute_manifest_hash(manifest):
        raise ValueError("hash 不匹配：manifest 内容已被篡改")


def manifest_summary(manifest: Mapping) -> dict:
    """返回可读摘要（各数据集题目数），用于生成脚本打印。"""
    return {
        d: manifest["datasets"][d]["count"] for d in MULTI_HOP_DATASETS if d in manifest["datasets"]
    }
