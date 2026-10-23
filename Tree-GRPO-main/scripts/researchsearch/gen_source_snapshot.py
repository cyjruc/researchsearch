#!/usr/bin/env python3
"""生成 docs/researchsearch/source-snapshot.md：计算仓库源码文件的 SHA-256 快照。

用法：
    python scripts/researchsearch/gen_source_snapshot.py

覆盖 Tree-GRPO-main 下除 .git / .pytest_cache / __pycache__ / .DS_Store 与
二进制图片（png/jpg 等）之外的所有文件，按相对路径排序输出。快照用于核对
「开发前基线 vs 当前内容」，不包含凭据、语料、索引或模型权重。
"""

from __future__ import annotations

import datetime
import hashlib
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]  # Tree-GRPO-main
_OUTPUT = _REPO / "docs/researchsearch/source-snapshot.md"

# 排除目录（任意层级命中即排除）。
_EXCLUDE_DIRS = {".git", ".pytest_cache", "__pycache__"}
# 排除文件（按文件名）。
_EXCLUDE_NAMES = {".DS_Store"}
# 排除后缀（二进制图片等非源码资源）。
_EXCLUDE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".whl",
    ".zip", ".tar", ".gz", ".bz2",
}


def _should_include(path: pathlib.Path) -> bool:
    if path.name in _EXCLUDE_NAMES:
        return False
    if path.suffix.lower() in _EXCLUDE_SUFFIXES:
        return False
    return not any(part in _EXCLUDE_DIRS for part in path.parts)


def main() -> int:
    entries = []
    for path in sorted(_REPO.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(_REPO)
        if not _should_include(rel):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((str(rel), digest))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    lines = [
        "# ResearchSearch-RL 源码快照（SHA-256）",
        "",
        f"生成时间：{now}",
        "",
        f"文件总数：{len(entries)}",
        "",
        "排除规则：.git、.pytest_cache、__pycache__、.DS_Store、二进制图片等非源码资源。",
        "",
        "## 文件哈希清单",
        "",
        "```",
    ]
    for rel, digest in entries:
        lines.append(f"{digest}  {rel}")
    lines.append("```")
    lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {len(entries)} 个文件到 {_OUTPUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
