#!/usr/bin/env python3
"""离线评测入口：读逐题预测 JSON，输出 EM / F1 / Recall@K / 可映射比例报告，或两个实验的消融。

用法：
    # 单实验评测（Wiki dev 或四项测试集）
    python scripts/researchsearch/evaluate.py --predictions preds.json --out report.json

    # 奖励消融（E2 vs E1、E3 vs E2），逐题可追溯
    python scripts/researchsearch/evaluate.py --baseline e1.json --comparison e2.json \
        --metric em --label-a E1 --label-b E2 --out ablation.json

输入 JSON 结构见 ``verl/utils/reward_score/eval_runner.py`` 顶部注释（逐题预测对象
列表，或包一层 ``{"predictions": [...]}``）。评测核心由 ``eval_metrics`` 与
``evidence.compute_recall_at_k`` 提供，编排口径由 ``eval_runner`` 锁定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from verl.utils.reward_score.eval_metrics import (  # noqa: E402
    bootstrap_ci,
    em_match,
    f1_match,
    mean_scores,
    multi_hop_avg_em,
    pairwise_ablation,
)
from verl.utils.reward_score.eval_runner import evaluate_predictions, run_ablation  # noqa: E402
from verl.utils.reward_score.evidence import compute_recall_at_k  # noqa: E402


def _scorers() -> SimpleNamespace:
    """把评测核心函数打包为 ``eval_runner`` 需要的 ``scorers`` 命名空间。"""
    return SimpleNamespace(
        em_match=em_match,
        f1_match=f1_match,
        mean_scores=mean_scores,
        multi_hop_avg_em=multi_hop_avg_em,
        bootstrap_ci=bootstrap_ci,
        pairwise_ablation=pairwise_ablation,
        compute_recall_at_k=compute_recall_at_k,
    )


def _load(path: Path) -> list:
    """读取预测 JSON，兼容 ``[{...}]`` 与 ``{"predictions": [{...}]}`` 两种结构。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        return data["predictions"]
    raise ValueError(
        f"{path} 结构不识别：须为逐题预测列表 [{...}]，或 {{'predictions': [...]}}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="离线评测：EM/F1/Recall@K/可映射比例 + 两实验消融"
    )
    parser.add_argument("--predictions", type=Path, help="逐题预测 JSON（单实验评测）")
    parser.add_argument("--baseline", type=Path, help="消融基线预测 JSON")
    parser.add_argument("--comparison", type=Path, help="消融对照预测 JSON")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[5, 10], help="Recall@K 的 K（默认 5 10）"
    )
    parser.add_argument(
        "--metric", choices=["em", "f1"], default="em", help="消融指标（默认 em）"
    )
    parser.add_argument("--label-a", default="A", help="消融基线标签")
    parser.add_argument("--label-b", default="B", help="消融对照标签")
    parser.add_argument("--out", type=Path, help="输出 JSON 路径（默认打印到 stdout）")
    args = parser.parse_args()

    scorers = _scorers()
    if args.predictions:
        report = evaluate_predictions(
            _load(args.predictions), scorers=scorers, k_values=args.k
        )
    elif args.baseline and args.comparison:
        report = run_ablation(
            _load(args.baseline),
            _load(args.comparison),
            scorers=scorers,
            label_a=args.label_a,
            label_b=args.label_b,
            metric=args.metric,
        )
    else:
        parser.error("须指定 --predictions，或同时指定 --baseline 与 --comparison")

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
