#!/usr/bin/env python3
"""逐叶证据接线离线检查（T02/T05 真实 rollout 后验证，方案 §4.4）。

读 generation_ts 在 EVIDENCE_EXPORT_DIR 门控下导出的 leaf_evidence.json，校验：
- 逐项对应：同一 uid（同一题）的所有叶子 gold(supporting_doc_ids) 一致，不串题；
- 覆盖率非零：至少一片叶子 visible_doc_ids ∩ supporting_doc_ids 非空；
- 两题 gold 不同：不同 uid 的 supporting_doc_ids 确实不同（样本确实混合）。

用法：
    python scripts/researchsearch/check_evidence_wiring.py --export-dir /path/to/evidence_export

任意一项失败退出码非 0。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_rows(export_dir: str):
    path = Path(export_dir) / "leaf_evidence.json"
    return json.loads(path.read_text(encoding="utf-8"))


def check_wiring(rows):
    """返回失败项列表（空 = 全通过）。"""
    failures = []

    if not rows:
        return ["leaf_evidence.json 为空，无任何叶子"]

    # 1. 逐项对应：同一 uid 的 gold 必须一致（不串题）。
    gold_by_uid = {}
    for r in rows:
        uid = r.get("uid")
        gold = tuple(sorted(r.get("supporting_doc_ids") or []))
        if uid in gold_by_uid and gold_by_uid[uid] != gold:
            failures.append(f"uid={uid} 的 gold 不一致：{gold_by_uid[uid]} vs {gold}")
        gold_by_uid[uid] = gold

    # 2. 覆盖率非零：至少一片叶子 visible ∩ gold 非空。
    coverage_leaves = 0
    for r in rows:
        visible = set(r.get("visible_doc_ids") or [])
        gold = set(r.get("supporting_doc_ids") or [])
        if visible & gold:
            coverage_leaves += 1
    if coverage_leaves == 0:
        failures.append("所有叶子 visible_doc_ids ∩ supporting_doc_ids 均为空（覆盖率零）")

    # 3. 两题 gold 不同：不同 uid 的 gold 集合数 ≥ 2（样本确实混合）。
    distinct_golds = {g for g in gold_by_uid.values() if g}
    if len(distinct_golds) < 2:
        failures.append(f"不同 uid 的 gold 集合数 < 2（实际 {len(distinct_golds)}），无法证明两题混合")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="逐叶证据接线离线检查")
    ap.add_argument("--export-dir", required=True)
    args = ap.parse_args()

    rows = load_rows(args.export_dir)
    print(f"已读 {len(rows)} 片叶子")
    failures = check_wiring(rows)
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: 逐项对应 / 覆盖率非零 / 两题 gold 不同 均通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
