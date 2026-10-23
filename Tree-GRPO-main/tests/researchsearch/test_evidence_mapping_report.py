"""证据映射报告脚本测试（第 2 周方案 §3.2 交付）。

锁定四态统计、语料缺失/歧义案例导出与人工核对样本抽样。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load():
    spec = importlib.util.spec_from_file_location(
        "evidence_mapping_report",
        _REPO / "scripts" / "researchsearch" / "evidence_mapping_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


report = _load()


def test_analyze_source_counts(tmp_path):
    d = tmp_path / "hotpotqa"
    d.mkdir()
    # 样本 a：Paris 唯一映射 → mapped；样本 b：Paris + Ghost（缺失）→ partial。
    (d / "train.jsonl").write_text(
        json.dumps(
            {"id": "a", "question": "Q1?", "golden_answers": ["A"],
             "metadata": {"supporting_facts": {"title": ["Paris"], "sent_id": [0]}}}
        )
        + "\n"
        + json.dumps(
            {"id": "b", "question": "Q2?", "golden_answers": ["B"],
             "metadata": {"supporting_facts": {"title": ["Paris", "Ghost"], "sent_id": [0, 0]}}}
        )
        + "\n",
        encoding="utf-8",
    )
    title_map = {"Paris": ["wiki:Paris"]}

    r = report.analyze_source(
        str(tmp_path), "hotpotqa", "train", title_map, seed=0, review_per_source=5
    )

    assert r["counts"] == {"mapped": 1, "partial": 1, "unmapped": 0, "not_available": 0}
    assert r["total"] == 2
    assert r["missing_title_counts"] == 1
    assert r["missing_title_examples"][0][0] == "Ghost"
    assert r["ambiguous_title_counts"] == 0
    # 两个样本都有支持事实 → 全部纳入人工核对池。
    assert len(r["review_samples"]) == 2


def test_analyze_source_skips_no_fact_from_review(tmp_path):
    d = tmp_path / "nq"
    d.mkdir()
    # nq 无证据字段 → not_available，且不纳入人工核对池。
    (d / "train.jsonl").write_text(
        json.dumps({"id": "a", "question": "Q1?", "golden_answers": ["A"]}) + "\n",
        encoding="utf-8",
    )
    r = report.analyze_source(str(tmp_path), "nq", "train", {}, seed=0, review_per_source=5)
    assert r["counts"] == {"mapped": 0, "partial": 0, "unmapped": 0, "not_available": 1}
    assert r["review_samples"] == []
