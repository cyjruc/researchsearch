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
"""评测 runner 核心：逐题预测的 schema 校验与评测编排（架构 §4 / 方案 §7 Day 7）。

本模块只做「编排」，不做「判定」：EM / F1 / 宏平均 / Multi-Hop Avg EM / bootstrap /
消融由 ``eval_metrics`` 提供，Recall@K 由 ``evidence.compute_recall_at_k`` 提供。
这些评测核心以 ``scorers`` 命名空间注入，使本模块保持零 verl 依赖，可在无 GPU /
无 tensordict 环境单文件加载测试（与 ``eval_metrics`` / ``evidence`` 同一约定）。

逐题预测 JSON 的 schema（每题一个对象）：

    {
      "question_id": "hotpotqa:dev:123",   # 必填，唯一，消融配对键
      "dataset": "hotpotqa",               # 必填，data_source（nq/hotpotqa/2wiki/musique/bamboogle）
      "prediction": "Paris",               # 必填，模型预测答案
      "golden_answers": ["Paris", "City of Light"],  # 必填，gold 答案或别名列表
      "retrieved_doc_ids": ["wiki:1", "wiki:2"],      # 可选，检索返回顺序（Recall@K 用）
      "visible_doc_ids": ["wiki:1"],                  # 可选，模型可见集合（证据覆盖率用）
      "gold_doc_ids": ["wiki:1", "wiki:9"],           # 可选，金标准支持文档
      "evidence_status": "mapped"         # 可选，默认 not_available
    }

锁定的口径（对应门禁 E06 / E07）：

- EM 独立于训练奖励：只读 ``prediction`` 与 ``golden_answers``，不读任何 reward /
  format 字段。
- Recall@K 与证据集合分离：Recall@K 读 ``retrieved_doc_ids``（检索返回集合），
  证据覆盖率读 ``visible_doc_ids``（模型可见集合），两字段不混用。
- Multi-Hop Avg EM 四项等权：缺数据集时在报告中如实标注 ``missing``，不静默沿用
  前三项（架构 §2 禁止只评前三项后沿用四项平均）。
- 消融逐题可追溯：按 ``question_id`` 配对，题目集不一致如实暴露 only_a / only_b。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

# 与 eval_metrics.MULTI_HOP_DATASETS 一致：Multi-Hop Avg EM 固定四项。
MULTI_HOP_DATASETS = ("hotpotqa", "2wiki", "musique", "bamboogle")

# 证据状态四态（与 build.py 的 EVIDENCE_STATUSES 一致）。
EVIDENCE_STATUSES = ("mapped", "partial", "unmapped", "not_available")

_REQUIRED_FIELDS = ("question_id", "dataset", "prediction", "golden_answers")


def _gold_list(golden_answers) -> list[str]:
    """把 gold 答案归一为列表（单个字符串视为单元素别名）。"""
    if isinstance(golden_answers, str):
        return [golden_answers]
    if golden_answers is None:
        return []
    return [str(g) for g in golden_answers]


def _id_list(item: dict, key: str) -> list[str]:
    """读取文档 ID 列表字段；缺省返回空列表，非序列类型报错。"""
    val = item.get(key)
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    raise ValueError(
        f"question_id={item.get('question_id')!r} 的 {key!r} 须为列表，收到 {type(val).__name__}"
    )


def validate_predictions(predictions: Sequence[dict]) -> list[dict]:
    """校验并规范化逐题预测，返回可安全下发的副本。

    缺必填字段、``question_id`` 为空或重复、字段类型错误、``evidence_status`` 不在
    四态内均抛 ``ValueError``——评测口径错误属于影响可信度的边界，不静默降级。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(predictions):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 条预测不是对象：{item!r}")
        for field in _REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(f"第 {i} 条预测缺少必填字段 {field!r}")

        qid = item["question_id"]
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"第 {i} 条预测 question_id 须为非空字符串：{qid!r}")
        if qid in seen:
            raise ValueError(f"question_id 重复：{qid!r}")
        seen.add(qid)

        dataset = item["dataset"]
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"question_id={qid!r} 的 dataset 须为非空字符串")

        prediction = item["prediction"]
        if not isinstance(prediction, str):
            raise ValueError(f"question_id={qid!r} 的 prediction 须为字符串")

        gold = _gold_list(item["golden_answers"])
        if not gold:
            raise ValueError(f"question_id={qid!r} 的 golden_answers 为空")

        evidence_status = item.get("evidence_status", "not_available")
        if evidence_status not in EVIDENCE_STATUSES:
            raise ValueError(
                f"question_id={qid!r} 的 evidence_status={evidence_status!r} "
                f"不在 {EVIDENCE_STATUSES}"
            )

        out.append(
            {
                "question_id": qid,
                "dataset": dataset,
                "prediction": prediction,
                "golden_answers": gold,
                "retrieved_doc_ids": _id_list(item, "retrieved_doc_ids"),
                "visible_doc_ids": _id_list(item, "visible_doc_ids"),
                "gold_doc_ids": _id_list(item, "gold_doc_ids"),
                "evidence_status": evidence_status,
            }
        )
    return out


def group_by_dataset(predictions: Sequence[dict]) -> dict[str, list[dict]]:
    """按 ``dataset`` 分组（保留输入顺序）。"""
    groups: dict[str, list[dict]] = {}
    for p in predictions:
        groups.setdefault(p["dataset"], []).append(p)
    return groups


def compute_em_by_dataset(predictions: Sequence[dict], em_match) -> dict[str, float]:
    """分数据集题级 EM 宏平均。"""
    out: dict[str, float] = {}
    for dataset, ps in group_by_dataset(predictions).items():
        scores = [float(em_match(p["prediction"], p["golden_answers"])) for p in ps]
        out[dataset] = sum(scores) / len(scores)
    return out


def compute_f1_by_dataset(predictions: Sequence[dict], f1_match) -> dict[str, float]:
    """分数据集题级 token F1 宏平均。"""
    out: dict[str, float] = {}
    for dataset, ps in group_by_dataset(predictions).items():
        scores = [float(f1_match(p["prediction"], p["golden_answers"])) for p in ps]
        out[dataset] = sum(scores) / len(scores)
    return out


def compute_recall_by_dataset(
    predictions: Sequence[dict],
    compute_recall_at_k,
    k: int,
) -> dict[str, dict]:
    """分数据集 Recall@K 宏平均，只统计 ``gold_doc_ids`` 非空（有证据标注）的样本。

    Recall@K 读 ``retrieved_doc_ids``（检索返回集合），与证据奖励的
    ``visible_doc_ids``（模型可见集合）分离留档（架构 §4）。
    """
    out: dict[str, dict] = {}
    for dataset, ps in group_by_dataset(predictions).items():
        recalls = []
        for p in ps:
            if not p["gold_doc_ids"]:
                continue
            result = compute_recall_at_k(p["gold_doc_ids"], p["retrieved_doc_ids"], k)
            recalls.append(result["recall"])
        out[dataset] = {
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "num_annotated": len(recalls),
        }
    return out


def compute_mappable_ratio(predictions: Sequence[dict]) -> dict:
    """证据可映射比例：``mapped / (mapped + partial + unmapped)``。

    ``not_available``（无金标准证据标注）不进分母；分母为 0 时比例记 0.0。
    返回四态计数、有标注样本数与比例，供 §6.2 单独报告 partial。
    """
    counts = {s: 0 for s in EVIDENCE_STATUSES}
    for p in predictions:
        counts[p["evidence_status"]] += 1
    annotated = counts["mapped"] + counts["partial"] + counts["unmapped"]
    ratio = counts["mapped"] / annotated if annotated else 0.0
    return {"counts": counts, "annotated": annotated, "ratio": ratio}


def evaluate_predictions(
    predictions: Sequence[dict],
    *,
    scorers: SimpleNamespace,
    k_values: Sequence[int] = (5, 10),
    multi_hop_datasets: Sequence[str] = MULTI_HOP_DATASETS,
) -> dict:
    """对逐题预测做完整评测，返回报告 dict。

    ``scorers`` 需含 ``em_match`` / ``f1_match`` / ``mean_scores`` /
    ``multi_hop_avg_em`` / ``bootstrap_ci`` / ``compute_recall_at_k``。
    Multi-Hop Avg EM 只在 ``multi_hop_datasets`` 全部存在时计算，否则置 None 并
    如实标注 ``multi_hop_missing``。
    """
    preds = validate_predictions(predictions)

    em_by = compute_em_by_dataset(preds, scorers.em_match)
    f1_by = compute_f1_by_dataset(preds, scorers.f1_match)

    all_em = [float(scorers.em_match(p["prediction"], p["golden_answers"])) for p in preds]
    all_f1 = [float(scorers.f1_match(p["prediction"], p["golden_answers"])) for p in preds]

    present = set(em_by)
    missing = [d for d in multi_hop_datasets if d not in present]
    multi_hop_avg = scorers.multi_hop_avg_em(em_by) if not missing else None

    return {
        "num_predictions": len(preds),
        "datasets": sorted(present),
        "em": {
            "by_dataset": em_by,
            "overall": scorers.mean_scores(all_em),
            "ci95": list(scorers.bootstrap_ci(all_em)),
        },
        "f1": {
            "by_dataset": f1_by,
            "overall": scorers.mean_scores(all_f1),
        },
        "multi_hop_avg_em": multi_hop_avg,
        "multi_hop_missing": missing,
        "recall": {
            k: compute_recall_by_dataset(preds, scorers.compute_recall_at_k, k)
            for k in k_values
        },
        "mappable": compute_mappable_ratio(preds),
    }


def run_ablation(
    preds_a: Sequence[dict],
    preds_b: Sequence[dict],
    *,
    scorers: SimpleNamespace,
    label_a: str = "A",
    label_b: str = "B",
    metric: str = "em",
) -> dict:
    """两个实验按 ``question_id`` 配对做消融，逐题差异可追溯。

    ``metric`` 为 ``em`` 或 ``f1``；逐题得分经规范化后交给 ``scorers.pairwise_ablation``，
    题目集不一致时其 ``only_a`` / ``only_b`` 字段如实暴露单侧题目。
    """
    if metric not in ("em", "f1"):
        raise ValueError(f"metric 须为 em/f1，收到 {metric!r}")
    score_fn = scorers.em_match if metric == "em" else scorers.f1_match

    a = {
        p["question_id"]: float(score_fn(p["prediction"], p["golden_answers"]))
        for p in validate_predictions(preds_a)
    }
    b = {
        p["question_id"]: float(score_fn(p["prediction"], p["golden_answers"]))
        for p in validate_predictions(preds_b)
    }
    return scorers.pairwise_ablation(a, b, label_a=label_a, label_b=label_b)
