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
"""从冻结 Wiki 语料构建 title → doc_id 映射表（title 级，第 2 周方案 §3.2）。

证据奖励采用 **title 级文档** 粒度（架构 §2.1：文档级证据覆盖，不代表句子级
证据充分性）：一篇维基文章（title）就是一个文档，``doc_id = wiki:<title>``。
wiki-18 语料是段落级切分——同一 title 会被拆成多个段落行，每行 ``contents``
首行都重复该 title。这些 chunk 只是检索/索引粒度，必须归并为同一文档，而不是
当作"同名歧义"。

输出 JSON：``{"<title>": ["wiki:<title>"], ...}``，仅含语料中实际存在的 title。
该表注入 ``build.py --title-docid-map``，供 ``evidence_map_fn`` 把 FlashRAG 的
supporting facts 标题对齐到稳定 doc_id，并核对金标准文档是否存在于冻结语料
（架构 §2.1：避免把语料缺失误判为搜索策略失败）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract_title_from_contents(contents: str) -> str:
    """从 wiki-18 的一行 ``contents`` 提取 title（首行，去掉包裹双引号）。

    wiki-18 的 ``contents`` 形如 ``"Title"\\n正文``：首行是 title，可能被字面
    双引号包裹。title 为空或缺失时返回空字符串，由调用方决定是否跳过。
    """
    if not contents:
        return ""
    first_line = contents.split("\n", 1)[0].strip()
    if len(first_line) >= 2 and first_line[0] == '"' and first_line[-1] == '"':
        first_line = first_line[1:-1]
    return first_line


def build_title_docid_map(corpus_path: str, *, doc_id_prefix: str = "wiki:") -> dict[str, list[str]]:
    """扫描 wiki-18 JSONL，返回 ``{title: [doc_id]}``（每 title 单元素）。

    同一 title 的多个段落行只记一次，doc_id 固定为 ``<prefix><title>``，实现
    title 级归并：段落级"一对多"不再产生歧义，而是同一文档的多个 chunk。
    """
    title_map: dict[str, list[str]] = {}
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            title = extract_title_from_contents(obj.get("contents", ""))
            if title:
                title_map.setdefault(title, [f"{doc_id_prefix}{title}"])
    return title_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 title→doc_id 映射表（title 级文档粒度）。")
    parser.add_argument("--corpus", required=True, help="wiki-18.jsonl 路径")
    parser.add_argument("--out", required=True, help="输出 title→doc_id 映射表 JSON")
    parser.add_argument("--doc-id-prefix", default="wiki:", help="doc_id 前缀（默认 wiki:）")
    args = parser.parse_args(argv)

    title_map = build_title_docid_map(args.corpus, doc_id_prefix=args.doc_id_prefix)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(title_map, fh, ensure_ascii=False)
    print(f"唯一 title 数: {len(title_map)}")
    print(f"映射表写入: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
