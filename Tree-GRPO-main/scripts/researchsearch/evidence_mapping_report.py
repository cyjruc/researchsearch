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
"""证据映射小样本报告（第 2 周方案 §3.2 交付）。

统计各数据源 supporting facts → doc_id 的四态分布（mapped/partial/unmapped/
not_available），导出语料缺失（missing）与同名歧义（ambiguous）案例，并抽样
≥20 条人工核对样本（含原始支持事实与映射结果）。

title 级 doc_id 下（架构 §2.1 / 第 2 周 §3.2 决策），同名 title 的多 chunk 已在
``build_title_docid_map.py`` 构建阶段归并为单一 doc_id，因此本报告的歧义案例应为
空；语料缺失 title 单独报告，避免把语料缺失误判为搜索策略失败。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main
sys.path.insert(0, str(_REPO))

from researchsearch.data.evidence_map import (  # noqa: E402
    extract_supporting_facts,
    map_supporting_facts,
)

EVIDENCE_STATUSES = ("mapped", "partial", "unmapped", "not_available")


def _iter_examples(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def analyze_source(
    flashrag_dir: str,
    source: str,
    split: str,
    title_map: dict,
    *,
    seed: int,
    review_per_source: int,
) -> dict:
    """统计单个数据源的证据映射四态，并抽样人工核对样本。"""
    path = Path(flashrag_dir) / source / f"{split}.jsonl"
    if not path.exists():
        raise ValueError(f"{path} 不存在")

    counts = {s: 0 for s in EVIDENCE_STATUSES}
    missing: dict[str, int] = {}
    ambiguous: dict[str, int] = {}
    review_pool: list[dict] = []

    for idx, example in _iter_examples(path):
        facts = extract_supporting_facts(example, source)
        result = map_supporting_facts(facts, title_map)
        counts[result["evidence_status"]] += 1
        for t in result["missing_titles"]:
            missing[t] = missing.get(t, 0) + 1
        for t in result["ambiguous_titles"]:
            ambiguous[t] = ambiguous.get(t, 0) + 1
        # 只把有支持事实的样本纳入人工核对（nq 无证据 → not_available，不核对）。
        if facts:
            review_pool.append(
                {
                    "source": source,
                    "source_index": idx,
                    "source_id": example.get("id"),
                    "question": example.get("question"),
                    "supporting_facts": facts,
                    "supporting_doc_ids": result["supporting_doc_ids"],
                    "evidence_status": result["evidence_status"],
                    "missing_titles": result["missing_titles"],
                    "ambiguous_titles": result["ambiguous_titles"],
                }
            )

    rng = random.Random(seed)
    review = rng.sample(review_pool, min(review_per_source, len(review_pool)))

    return {
        "source": source,
        "split": split,
        "counts": counts,
        "total": sum(counts.values()),
        "mappable_rate": counts["mapped"] / sum(counts.values()) if counts else 0.0,
        "missing_title_counts": len(missing),
        "missing_title_examples": sorted(missing.items(), key=lambda kv: -kv[1])[:20],
        "ambiguous_title_counts": len(ambiguous),
        "ambiguous_title_examples": sorted(ambiguous.items(), key=lambda kv: -kv[1])[:20],
        "review_samples": review,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="证据映射小样本报告（第 2 周 §3.2 交付）。")
    parser.add_argument("--flashrag-dir", required=True, help="FlashRAG 数据集根目录")
    parser.add_argument("--title-docid-map", required=True, help="title→doc_id 映射表 JSON")
    parser.add_argument("--sources", required=True, help="逗号分隔的 data_source 名")
    parser.add_argument("--split", default="train", help="split（train/dev）")
    parser.add_argument("--review-per-source", type=int, default=7, help="每源人工核对样本数")
    parser.add_argument("--seed", type=int, default=42, help="人工核对样本抽样种子")
    parser.add_argument("--out", required=True, help="报告输出 JSON")
    args = parser.parse_args(argv)

    with open(args.title_docid_map, "r", encoding="utf-8") as fh:
        title_map = json.load(fh)

    sources = [s for s in args.sources.split(",") if s]
    reports = [
        analyze_source(
            args.flashrag_dir, source, args.split, title_map,
            seed=args.seed, review_per_source=args.review_per_source,
        )
        for source in sources
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, ensure_ascii=False, indent=2)

    total_counts = {s: 0 for s in EVIDENCE_STATUSES}
    total_review = 0
    for r in reports:
        print(f"{r['source']}/{r['split']}: {r['counts']} (total={r['total']}, mapped率={r['mappable_rate']:.4f})")
        print(f"  missing title 数={r['missing_title_counts']}, ambiguous title 数={r['ambiguous_title_counts']}")
        for s in EVIDENCE_STATUSES:
            total_counts[s] += r["counts"][s]
        total_review += len(r["review_samples"])

    total = sum(total_counts.values())
    print(f"汇总: {total_counts} (total={total}, mapped率={total_counts['mapped'] / total if total else 0:.4f})")
    print(f"人工核对样本数: {total_review}")
    print(f"报告写入: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
