"""title→doc_id 映射表构建脚本测试（第 2 周方案 §3.2，title 级文档粒度）。

锁定两点：
- wiki-18 ``contents`` 首行 title 提取（含双引号包裹、空内容）。
- 同一 title 的多个段落行归并为同一文档（``doc_id = wiki:<title>``），
  不产生"同名歧义"。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_title_docid_map",
        _REPO / "scripts" / "researchsearch" / "build_title_docid_map.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def test_extract_title_strips_wrapping_quotes():
    assert mod.extract_title_from_contents('"Ed Wood"\nWood handled...') == "Ed Wood"


def test_extract_title_plain_first_line():
    assert mod.extract_title_from_contents("Paris\nCapital of France") == "Paris"


def test_extract_title_empty_content():
    assert mod.extract_title_from_contents("") == ""
    assert mod.extract_title_from_contents(None) == ""


def test_build_title_docid_map_merges_chunks_by_title(tmp_path):
    corpus = tmp_path / "wiki.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "contents": '"Ed Wood"\nchunk A'}),
                json.dumps({"id": 2, "contents": '"Ed Wood"\nchunk B'}),
                json.dumps({"id": 3, "contents": '"Paris"\nchunk C'}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    title_map = mod.build_title_docid_map(str(corpus))
    # 同一 title 的多个 chunk 归并为同一文档，无歧义
    assert title_map == {"Ed Wood": ["wiki:Ed Wood"], "Paris": ["wiki:Paris"]}


def test_build_title_docid_map_custom_prefix(tmp_path):
    corpus = tmp_path / "wiki.jsonl"
    corpus.write_text(json.dumps({"id": 1, "contents": '"Paris"\ntext'}) + "\n", encoding="utf-8")
    title_map = mod.build_title_docid_map(str(corpus), doc_id_prefix="doc:")
    assert title_map == {"Paris": ["doc:Paris"]}
