#!/usr/bin/env python3
"""生成 Multi-Hop Avg EM 测试集 manifest.json（预注册骨架，架构 §2 / 方案 §10 第 9 项）。

用法：
    python scripts/researchsearch/gen_eval_manifest.py --input datasets.json --out manifest.json \
        [--split test] [--data-version wiki-2026-01] [--filter-rules "..."]

输入 datasets.json：
    {
      "hotpotqa": [{"question_id": "hotpotqa:test:xxx", "golden_answers": ["..."], "source_id": "xxx"}],
      "2wiki": [...], "musique": [...], "bamboogle": [...]
    }

真实题目 ID / 数据版本待测试集文件核对后回填；本脚本只填骨架，四项数据集缺一项
即报错，不伪称已预注册。生成物含只读哈希，篡改题目/别名/评分器/版本会使校验失败。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load_eval_manifest():
    # eval_manifest 是纯标准库模块；用 spec_from_file_location 单文件加载，避免触发
    # verl 包 import，使本脚本在无语料 / 无 tensordict 环境同样可运行。
    path = _REPO / "verl/utils/reward_score/eval_manifest.py"
    spec = importlib.util.spec_from_file_location("researchsearch_eval_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    manifest_mod = _load_eval_manifest()
    parser = argparse.ArgumentParser(description="生成 Multi-Hop Avg EM 测试集 manifest.json")
    parser.add_argument("--input", type=Path, required=True, help="datasets JSON（{dataset: [条目]}）")
    parser.add_argument("--out", type=Path, required=True, help="输出 manifest.json")
    parser.add_argument("--split", default="test", help="评测 split（默认 test）")
    parser.add_argument("--data-version", default=None, help="统一数据版本字符串")
    parser.add_argument("--filter-rules", default="", help="筛选规则说明")
    parser.add_argument("--scorer-version", default="1", help="评分器版本")
    args = parser.parse_args()

    datasets = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = manifest_mod.build_eval_manifest(
        datasets,
        split=args.split,
        data_version=args.data_version,
        filter_rules=args.filter_rules,
        scorer_version=args.scorer_version,
    )
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {args.out}")
    print(f"题目数 {manifest_mod.manifest_summary(manifest)}")
    print(f"hash {manifest['hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
