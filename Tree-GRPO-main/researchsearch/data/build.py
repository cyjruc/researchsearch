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
"""统一数据构建：多源合并、规范化去重、全局唯一 index 与 manifest。

对齐《第 1 周工程闭环开发方案》§4.2 的统一样本结构。核心函数不依赖
datasets / torch，仅用标准库，便于在本机用 fixture 跑单元测试；数据源读取
（FlashRAG 目录）单独实现，目标机可用。

设计要点：
- ``extra_info.index`` 是拼接后的**全局唯一整数**训练/优势分组 ID，跨数据集
  不重置；原始行号另存为 ``source_index``。
- ``sample_id = "<data_source>:<source_id>"``，与 ``index`` 一一对应。
- 跨数据集按规范化问题去重；train/dev 之间重叠视为泄漏，从 dev 中移除。
- 支持文档元数据只写入 ``extra_info``，不进入模型可见的 ``prompt``。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# 证据状态枚举（方案 §4.2 要求区分 mapped / partial / unmapped / not_available）
EVIDENCE_STATUSES = ("mapped", "partial", "unmapped", "not_available")

DEFAULT_ABILITY = "fact-reasoning"
DEFAULT_ENVIRONMENT = "wiki"

# FlashRAG 数据源的 split 选择：train 用官方 train，dev 用官方 dev（独立划分）。
# 正式 test split 锁定，dev 数据源不得消费 test（方案 §4.1「正式测试集锁定」）。
_TRAIN_FLASHRAG_SPLIT = "train"
_DEV_FLASHRAG_SPLIT = "dev"

_UNWANTED_QUESTION_CHARS = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    """规范化问题文本，用于跨数据集去重与泄漏检测。

    小写、去首尾空白、压缩内部空白、去标点。这是保守的规范化：只做对去重
    有必要的变换，不改变语义（不词干化、不去停用词）。
    """
    text = question.strip().lower()
    text = _UNWANTED_QUESTION_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def question_key(question: str) -> str:
    """去重字典的稳定键：规范化文本本身即可读，这里保留为独立函数便于扩展。"""
    return normalize_question(question)


def make_prefix(question: str, template_type: str = "base") -> str:
    """构造 ReAct 提示前缀。只包含问题，不含答案与证据。"""
    if template_type == "base":
        return (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search> and it will return the top searched results between <information> "
            "and </information>. "
            "You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer inside "
            "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
            f"Question: {question}\n"
        )
    raise NotImplementedError(f"template_type={template_type!r}")


def _clean_question(question: str) -> str:
    q = question.strip()
    if q and q[-1] != "?":
        q += "?"
    return q


def _as_answer_list(golden_answers: Any) -> list[str]:
    if isinstance(golden_answers, str):
        return [golden_answers]
    return [str(a) for a in golden_answers]


def build_sample(
    *,
    data_source: str,
    question: str,
    golden_answers: Any,
    split: str,
    index: int,
    source_index: int,
    sample_id: str,
    source_revision: Optional[str] = None,
    environment: str = DEFAULT_ENVIRONMENT,
    supporting_doc_ids: Optional[Iterable[str]] = None,
    evidence_status: Optional[str] = None,
    corpus_version: Optional[str] = None,
    template_type: str = "base",
) -> dict:
    """构造一条统一 parquet 行（方案 §4.2 样本结构）。"""
    supporting = list(supporting_doc_ids) if supporting_doc_ids else []
    if evidence_status is None:
        evidence_status = "mapped" if supporting else "not_available"
    if evidence_status not in EVIDENCE_STATUSES:
        raise ValueError(f"evidence_status={evidence_status!r} 不在 {EVIDENCE_STATUSES}")

    extra_info = {
        "sample_id": sample_id,
        "index": index,
        "source_index": source_index,
        "split": split,
        "source_revision": source_revision,
        "environment": environment,
        "supporting_doc_ids": supporting,
        "evidence_status": evidence_status,
        "corpus_version": corpus_version,
    }
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": make_prefix(_clean_question(question), template_type=template_type)}],
        "ability": DEFAULT_ABILITY,
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": _as_answer_list(golden_answers)},
        },
        "extra_info": extra_info,
    }


def _record_sample_id(record: dict) -> str:
    source_id = record.get("source_id") or str(record.get("source_index"))
    return f"{record['data_source']}:{source_id}"


def _normalize_record(record: dict) -> dict:
    """补齐记录默认字段，供 build_dataset 使用。"""
    out = dict(record)
    if "source_index" not in out:
        raise ValueError("record 缺少 source_index 字段")
    if "question" not in out or "golden_answers" not in out:
        raise ValueError("record 缺少 question / golden_answers 字段")
    out.setdefault("sample_id", _record_sample_id(out))
    return out


def _deduplicate(records: list[dict], split: str) -> tuple[list[dict], list[dict]]:
    """按规范化问题去重，保留首次出现；返回 (去重后记录, 被移除记录)。"""
    seen: dict[str, dict] = {}
    removed: list[dict] = []
    for record in records:
        key = question_key(record["question"])
        if key in seen:
            removed.append({"sample_id": record["sample_id"], "split": split, "question": record["question"]})
        else:
            seen[key] = record
    return list(seen.values()), removed


def build_dataset(
    train_records: list[dict],
    dev_records: list[dict],
    *,
    start_index: int = 0,
    environment: str = DEFAULT_ENVIRONMENT,
    corpus_version: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """合并 train/dev，去重、分配全局唯一 index，构造 sample 与 manifest。

    参数：
        train_records / dev_records: 各为按固定 manifest 顺序排列的记录列表。
            每条记录至少含 ``data_source``、``source_index``、``question``、
            ``golden_answers``；可选 ``source_id``、``supporting_doc_ids``、
            ``evidence_status``、``source_revision``。
        start_index: 全局 index 起始值（train 从它开始，dev 紧接其后）。
        environment / corpus_version: 写入 extra_info 的语料级元数据。
        seed: 数据抽样种子，写入 manifest（方案 §4.1「逐条可追溯数据抽样种子」）。

    返回 dict：``train_samples``、``dev_samples``、``manifest``、``removed``。
    """
    train_records = [_normalize_record(r) for r in train_records]
    dev_records = [_normalize_record(r) for r in dev_records]

    # 1. 泄漏检测：train/dev 规范化问题重叠时，从 dev 移除（保留在 train）。
    train_keys = {question_key(r["question"]) for r in train_records}
    leaked: list[dict] = []
    kept_dev: list[dict] = []
    for record in dev_records:
        if question_key(record["question"]) in train_keys:
            leaked.append({"sample_id": record["sample_id"], "split": "dev", "question": record["question"]})
        else:
            kept_dev.append(record)

    # 2. 各 split 内部按规范化问题去重。
    train_unique, train_dup = _deduplicate(train_records, "train")
    dev_unique, dev_dup = _deduplicate(kept_dev, "dev")

    # 3. 按固定顺序（train 在前，dev 在后）分配全局唯一连续 index。
    train_samples = []
    dev_samples = []
    manifest_entries = []
    cursor = start_index

    def _build(records, split):
        nonlocal cursor
        samples = []
        for record in records:
            sample = build_sample(
                data_source=record["data_source"],
                question=record["question"],
                golden_answers=record["golden_answers"],
                split=split,
                index=cursor,
                source_index=record["source_index"],
                sample_id=record["sample_id"],
                source_revision=record.get("source_revision"),
                environment=environment,
                supporting_doc_ids=record.get("supporting_doc_ids"),
                evidence_status=record.get("evidence_status"),
                corpus_version=corpus_version,
            )
            manifest_entries.append(
                {
                    "sample_id": record["sample_id"],
                    "index": cursor,
                    "data_source": record["data_source"],
                    "source_id": record.get("source_id"),
                    "source_index": record["source_index"],
                    "split": split,
                    "evidence_status": sample["extra_info"]["evidence_status"],
                    "question": record["question"],
                }
            )
            samples.append(sample)
            cursor += 1
        return samples

    train_samples = _build(train_unique, "train")
    dev_samples = _build(dev_unique, "dev")

    manifest = {
        "schema_version": 1,
        "environment": environment,
        "corpus_version": corpus_version,
        "start_index": start_index,
        "seed": seed,
        "train": {"count": len(train_samples), "index_range": [start_index, start_index + len(train_samples)]},
        "dev": {
            "count": len(dev_samples),
            "index_range": [start_index + len(train_samples), start_index + len(train_samples) + len(dev_samples)],
        },
        "entries": manifest_entries,
    }

    removed = {
        "train_dev_leak": leaked,
        "train_duplicates": train_dup,
        "dev_duplicates": dev_dup,
    }

    return {
        "train_samples": train_samples,
        "dev_samples": dev_samples,
        "manifest": manifest,
        "removed": removed,
    }


# ---------------------------------------------------------------------------
# 数据源读取（与核心逻辑解耦）
# ---------------------------------------------------------------------------


def load_jsonl_source(
    path: str,
    *,
    data_source: str,
    source_revision: Optional[str] = None,
) -> list[dict]:
    """从本地 JSONL 读取一个数据源（本机 smoke / fixture）。

    每行一个 JSON 对象，至少含 ``question`` 与 ``golden_answers``；可选
    ``source_id``、``supporting_doc_ids``、``evidence_status``。
    ``source_index`` 默认按行号生成，也可由行内字段显式覆盖。
    """
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj or "golden_answers" not in obj:
                raise ValueError(f"{path} 第 {idx} 行缺少 question / golden_answers")
            record = {
                "data_source": data_source,
                "source_id": obj.get("source_id", str(idx)),
                "source_index": obj.get("source_index", idx),
                "question": obj["question"],
                "golden_answers": obj["golden_answers"],
                "supporting_doc_ids": obj.get("supporting_doc_ids"),
                "evidence_status": obj.get("evidence_status"),
                "source_revision": obj.get("source_revision", source_revision),
            }
            records.append(record)
    return records


def load_flashrag_source(
    dataset_dir: str,
    data_source: str,
    split: str,
    *,
    source_revision: Optional[str] = None,
    evidence_map_fn=None,
) -> list[dict]:
    """从 FlashRAG 目录读取一个数据源（JSONL 直存格式）。

    FlashRAG 数据集（RUC-NLPIR/FlashRAG_datasets）以 JSONL 存储：每个
    ``data_source`` 一个子目录，含 ``train.jsonl`` / ``dev.jsonl``（部分源还有
    ``test.jsonl``）。``split`` 对应的 JSONL 必须存在，否则抛出 ``ValueError``
    ——不自动回退到其他 split，避免正式生成数据时 train/dev 隔离失效。

    ``evidence_map_fn(example, idx)`` 返回 ``{"supporting_doc_ids": [...],
    "evidence_status": "..."}``，用于把各数据集的结构化支持事实映射到统一
    doc_id。未提供时证据字段留空（evidence_status=not_available）。
    """
    path = Path(dataset_dir) / data_source / f"{split}.jsonl"
    if not path.exists():
        raise ValueError(
            f"data_source={data_source!r} 没有 split={split!r} 的 JSONL（{path} 不存在）。"
            "为保持 train/dev 隔离，不自动回退到其他 split。"
        )

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            evidence = evidence_map_fn(example, idx) if evidence_map_fn else {}
            records.append(
                {
                    "data_source": data_source,
                    "source_id": str(example.get("id", idx)),
                    "source_index": idx,
                    "question": example["question"],
                    "golden_answers": example["golden_answers"],
                    "supporting_doc_ids": evidence.get("supporting_doc_ids"),
                    "evidence_status": evidence.get("evidence_status"),
                    "source_revision": source_revision,
                }
            )
    return records


def _split_sources(spec: Optional[str]) -> list[str]:
    return [s for s in (spec or "").split(",") if s]


def _sample_records(records: list[dict], n: Optional[int], *, seed: int) -> list[dict]:
    """按 ``seed`` 确定性抽样 ``n`` 条；不足 ``n`` 全取。抽后按 ``source_index``
    恢复原始顺序，保证 manifest 顺序稳定、可复现（方案 §3.1 抽样种子）。"""
    if n is None or n >= len(records):
        return records
    rng = random.Random(seed)
    sampled = rng.sample(records, n)
    return sorted(sampled, key=lambda r: r["source_index"])


def _write_split(out_dir: Path, samples: list[dict], split: str) -> None:
    if not samples:
        return
    import pandas as pd  # 延迟导入，核心逻辑不依赖

    pd.DataFrame(samples).to_parquet(out_dir / f"{split}.parquet", index=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建统一 Wiki 数据（去重 + 全局唯一 index + manifest）。")
    parser.add_argument("--train-jsonl", default=None, help="逗号分隔的本地 JSONL（本机 smoke / fixture）")
    parser.add_argument("--dev-jsonl", default=None, help="逗号分隔的本地 JSONL（本机 smoke / fixture）")
    parser.add_argument("--train-flashrag", default=None, help="逗号分隔的 data_source 名（目标机，配合 --flashrag-dir）")
    parser.add_argument("--dev-flashrag", default=None, help="逗号分隔的 data_source 名（目标机，配合 --flashrag-dir）")
    parser.add_argument("--flashrag-dir", default=None, help="FlashRAG 数据集根目录")
    parser.add_argument("--out-dir", required=True, help="输出目录（写 train.parquet / dev.parquet / manifest.json）")
    parser.add_argument("--start-index", type=int, default=0, help="全局 index 起始值")
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--corpus-version", default=None)
    parser.add_argument("--seed", type=int, default=None, help="数据抽样种子（写入 manifest，可追溯）")
    parser.add_argument("--train-sample", default=None, help="逗号分隔每源抽样数（与 --train-flashrag 顺序对应，需 --seed）")
    parser.add_argument("--dev-sample", default=None, help="逗号分隔每源抽样数（与 --dev-flashrag 顺序对应，需 --seed）")
    parser.add_argument("--title-docid-map", default=None, help="title→doc_id 映射表 JSON（{title: [doc_id]}，注入 evidence_map_fn）")
    parser.add_argument("--source-revision", default=None, help="统一来源版本（所有来源共用）")
    args = parser.parse_args(argv)

    title_map: dict = {}
    if args.title_docid_map:
        with open(args.title_docid_map, "r", encoding="utf-8") as fh:
            title_map = json.load(fh)

    train_sample = [int(x) for x in _split_sources(args.train_sample)] if args.train_sample else []
    dev_sample = [int(x) for x in _split_sources(args.dev_sample)] if args.dev_sample else []
    if (train_sample or dev_sample) and args.seed is None:
        parser.error("抽样（--train-sample / --dev-sample）必须配合 --seed 以保持可复现")

    train_records: list[dict] = []
    dev_records: list[dict] = []

    for source in _split_sources(args.train_jsonl):
        train_records.extend(load_jsonl_source(source, data_source=Path(source).stem, source_revision=args.source_revision))
    for source in _split_sources(args.dev_jsonl):
        dev_records.extend(load_jsonl_source(source, data_source=Path(source).stem, source_revision=args.source_revision))

    if args.train_flashrag or args.dev_flashrag:
        if not args.flashrag_dir:
            parser.error("--flashrag-dir 必须与 --train-flashrag / --dev-flashrag 一起提供")
        from researchsearch.data import evidence_map  # 延迟导入，保持核心逻辑零重依赖

        def _evidence_fn(data_source):
            if not title_map:
                return None
            return evidence_map.make_evidence_map_fn(title_map, data_source=data_source)

        for i, data_source in enumerate(_split_sources(args.train_flashrag)):
            recs = load_flashrag_source(
                args.flashrag_dir, data_source, _TRAIN_FLASHRAG_SPLIT,
                source_revision=args.source_revision,
                evidence_map_fn=_evidence_fn(data_source),
            )
            n = train_sample[i] if i < len(train_sample) else None
            train_records.extend(_sample_records(recs, n, seed=args.seed))
        for i, data_source in enumerate(_split_sources(args.dev_flashrag)):
            recs = load_flashrag_source(
                args.flashrag_dir, data_source, _DEV_FLASHRAG_SPLIT,
                source_revision=args.source_revision,
                evidence_map_fn=_evidence_fn(data_source),
            )
            n = dev_sample[i] if i < len(dev_sample) else None
            dev_records.extend(_sample_records(recs, n, seed=args.seed))

    if not train_records and not dev_records:
        parser.error("至少提供一个数据源（--train-jsonl / --dev-jsonl / --train-flashrag / --dev-flashrag）")

    result = build_dataset(
        train_records,
        dev_records,
        start_index=args.start_index,
        environment=args.environment,
        corpus_version=args.corpus_version,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_split(out_dir, result["train_samples"], "train")
    _write_split(out_dir, result["dev_samples"], "dev")
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(result["manifest"], fh, ensure_ascii=False, indent=2)

    removed = result["removed"]
    print(f"train: {len(result['train_samples'])} 条, dev: {len(result['dev_samples'])} 条")
    print(f"train/dev 泄漏移除: {len(removed['train_dev_leak'])} 条")
    print(f"train 内部去重移除: {len(removed['train_duplicates'])} 条, dev 内部去重移除: {len(removed['dev_duplicates'])} 条")
    print(f"manifest 写入: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
