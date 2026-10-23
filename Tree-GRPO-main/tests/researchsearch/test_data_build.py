"""数据层单元测试：全局唯一 index、去重、泄漏检测、证据字段与 prompt 隔离。

对齐方案 T01（数据隔离与分组 ID）以及 §4.2 的样本结构约定。
"""

import json

import pytest

from researchsearch.data.build import (
    EVIDENCE_STATUSES,
    _DEV_FLASHRAG_SPLIT,
    _TRAIN_FLASHRAG_SPLIT,
    _sample_records,
    build_dataset,
    build_sample,
    load_flashrag_source,
    normalize_question,
)


def make_record(
    data_source,
    source_index,
    question,
    answers,
    source_id=None,
    supporting_doc_ids=None,
    evidence_status=None,
):
    return {
        "data_source": data_source,
        "source_id": source_id if source_id is not None else f"{data_source}-{source_index}",
        "source_index": source_index,
        "question": question,
        "golden_answers": answers,
        "supporting_doc_ids": supporting_doc_ids,
        "evidence_status": evidence_status,
    }


def _indexes(samples):
    return [s["extra_info"]["index"] for s in samples]


def test_global_index_unique_contiguous_and_disjoint():
    train = [
        make_record("hotpotqa", 0, "Who is the author of Harry Potter?", ["J.K. Rowling"]),
        make_record("hotpotqa", 1, "What is the capital of France?", ["Paris"]),
        make_record("2wiki", 0, "Who invented the telephone?", ["Alexander Graham Bell"]),
        make_record("2wiki", 1, "What is the tallest mountain?", ["Everest"]),
    ]
    dev = [
        make_record("hotpotqa", 0, "What is the largest ocean?", ["Pacific"]),
    ]
    result = build_dataset(train, dev, start_index=0)

    train_idx = _indexes(result["train_samples"])
    dev_idx = _indexes(result["dev_samples"])

    # 全局唯一、从 start_index 连续
    all_idx = train_idx + dev_idx
    assert all_idx == list(range(len(all_idx)))
    # train/dev 无交集（T01）
    assert set(train_idx).isdisjoint(dev_idx)
    assert dev_idx == [4]


def test_cross_source_same_row_number_gets_distinct_index():
    # 跨数据集同原始行号不能串 uid 组（T01 的核心回归）
    train = [
        make_record("nq", 0, "Question A?", ["A"]),
        make_record("hotpotqa", 0, "Question B?", ["B"]),
    ]
    result = build_dataset(train, [], start_index=0)
    assert _indexes(result["train_samples"]) == [0, 1]
    sample_ids = [s["extra_info"]["sample_id"] for s in result["train_samples"]]
    assert len(set(sample_ids)) == 2


def test_sample_id_unique_and_matches_index():
    train = [
        make_record("hotpotqa", 0, "Question A?", ["A"]),
        make_record("hotpotqa", 1, "Question B?", ["B"]),
    ]
    result = build_dataset(train, [], start_index=7)
    samples = result["train_samples"]
    ids = [s["extra_info"]["sample_id"] for s in samples]
    assert len(set(ids)) == len(samples)
    # index ↔ sample_id 一一对应（manifest 也应如此）
    for s in samples:
        assert s["extra_info"]["index"] in (7, 8)


def test_dedup_within_split():
    train = [
        make_record("hotpotqa", 0, "What is the capital of France?", ["Paris"]),
        # 规范化后与上一条相同（大小写/标点差异）
        make_record("hotpotqa", 1, "what is the capital of france", ["Paris"]),
        make_record("hotpotqa", 2, "Who wrote Hamlet?", ["Shakespeare"]),
    ]
    result = build_dataset(train, [], start_index=0)
    assert len(result["train_samples"]) == 2
    assert len(result["removed"]["train_duplicates"]) == 1


def test_train_dev_leak_removed_from_dev():
    train = [make_record("hotpotqa", 0, "What is the capital of France?", ["Paris"])]
    dev = [make_record("hotpotqa", 0, "What is the capital of France?", ["Paris"])]
    result = build_dataset(train, dev, start_index=0)
    assert len(result["train_samples"]) == 1
    assert len(result["dev_samples"]) == 0
    assert len(result["removed"]["train_dev_leak"]) == 1


def test_normalization_is_case_and_punctuation_insensitive():
    a = "What is the capital of France?"
    b = "  what is the capital of france  "
    assert normalize_question(a) == normalize_question(b)


def test_prompt_excludes_answers_and_evidence():
    record = make_record(
        "hotpotqa",
        0,
        "Who wrote the novel 1984?",
        ["George Orwell"],
        supporting_doc_ids=["wiki:doc_george_orwell"],
        evidence_status="mapped",
    )
    sample = build_sample(
        data_source=record["data_source"],
        question=record["question"],
        golden_answers=record["golden_answers"],
        split="train",
        index=0,
        source_index=record["source_index"],
        sample_id=f"{record['data_source']}:{record['source_id']}",
        supporting_doc_ids=record["supporting_doc_ids"],
        evidence_status=record["evidence_status"],
    )
    prompt_text = sample["prompt"][0]["content"]
    # 答案与支持文档 ID 都不得进入模型可见的 prompt
    assert "George Orwell" not in prompt_text
    assert "wiki:doc_george_orwell" not in prompt_text
    # 但应写入 extra_info，供证据奖励旁路读取
    assert sample["extra_info"]["supporting_doc_ids"] == ["wiki:doc_george_orwell"]
    assert sample["extra_info"]["evidence_status"] == "mapped"


def test_evidence_status_defaults():
    # 无证据 → not_available；有支持文档但未显式给 status → mapped
    no_evidence = make_record("hotpotqa", 0, "Question A?", ["A"])
    with_evidence = make_record("hotpotqa", 1, "Question B?", ["B"], supporting_doc_ids=["wiki:d1"])
    result = build_dataset([no_evidence, with_evidence], [], start_index=0)
    statuses = [s["extra_info"]["evidence_status"] for s in result["train_samples"]]
    assert statuses == ["not_available", "mapped"]


def test_invalid_evidence_status_raises():
    with pytest.raises(ValueError):
        build_sample(
            data_source="hotpotqa",
            question="Question A?",
            golden_answers=["A"],
            split="train",
            index=0,
            source_index=0,
            sample_id="hotpotqa:0",
            evidence_status="bogus",
        )


def test_manifest_consistent_with_samples():
    train = [
        make_record("hotpotqa", 0, "Question A?", ["A"]),
        make_record("2wiki", 0, "Question B?", ["B"]),
    ]
    dev = [make_record("musique", 0, "Question C?", ["C"])]
    result = build_dataset(train, dev, start_index=0)

    manifest = result["manifest"]
    assert manifest["train"]["count"] == 2
    assert manifest["dev"]["count"] == 1
    assert manifest["train"]["index_range"] == [0, 2]
    assert manifest["dev"]["index_range"] == [2, 3]

    entries = manifest["entries"]
    assert len(entries) == 3
    # entries 的 index/sample_id 与样本一一对应
    assert [e["index"] for e in entries] == [0, 1, 2]
    assert [e["sample_id"] for e in entries] == [
        s["extra_info"]["sample_id"] for s in result["train_samples"] + result["dev_samples"]
    ]


def test_manifest_records_seed():
    train = [make_record("hotpotqa", 0, "Question A?", ["A"])]
    result = build_dataset(train, [], start_index=0, seed=42)
    assert result["manifest"]["seed"] == 42


def test_manifest_seed_defaults_none():
    result = build_dataset([], [], start_index=0)
    assert result["manifest"]["seed"] is None


def test_evidence_statuses_enum_unchanged():
    # 防止意外改动破坏方案 §4.2 约定的四种状态
    assert EVIDENCE_STATUSES == ("mapped", "partial", "unmapped", "not_available")


# --- FlashRAG split 选择（方案 §4.1「正式测试集锁定」） ------------------------


def test_flashrag_split_constants_never_use_test():
    # dev 数据源只能取官方 dev，绝不消费 test split。
    assert _TRAIN_FLASHRAG_SPLIT == "train"
    assert _DEV_FLASHRAG_SPLIT == "dev"


def test_load_flashrag_reads_jsonl(tmp_path):
    # FlashRAG 是 JSONL 直存：<dir>/<data_source>/<split>.jsonl。
    d = tmp_path / "hotpotqa"
    d.mkdir()
    (d / "dev.jsonl").write_text(
        json.dumps({"id": "d0", "question": "dev question?", "golden_answers": ["d"]}) + "\n",
        encoding="utf-8",
    )
    records = load_flashrag_source(str(tmp_path), "hotpotqa", _DEV_FLASHRAG_SPLIT)
    assert [r["question"] for r in records] == ["dev question?"]
    assert records[0]["source_id"] == "d0"


def test_load_flashrag_missing_split_raises(tmp_path):
    # split 对应 JSONL 不存在时直接报错，不悄悄回退到其他 split（保持 train/dev 隔离）。
    d = tmp_path / "hotpotqa"
    d.mkdir()
    (d / "train.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="没有 split"):
        load_flashrag_source(str(tmp_path), "hotpotqa", _DEV_FLASHRAG_SPLIT)


def test_load_flashrag_source_applies_evidence_map_fn(tmp_path):
    # evidence_map_fn(example, idx) 的返回字段写入 supporting_doc_ids / evidence_status。
    d = tmp_path / "hotpotqa"
    d.mkdir()
    example = {
        "id": "d0",
        "question": "dev question?",
        "golden_answers": ["d"],
        "metadata": {"supporting_facts": {"title": ["Paris"], "sent_id": [0]}},
    }
    (d / "dev.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")

    def evidence_map_fn(example, idx):
        return {"supporting_doc_ids": ["wiki:Paris"], "evidence_status": "mapped"}

    records = load_flashrag_source(
        str(tmp_path), "hotpotqa", _DEV_FLASHRAG_SPLIT, evidence_map_fn=evidence_map_fn
    )
    assert records[0]["supporting_doc_ids"] == ["wiki:Paris"]
    assert records[0]["evidence_status"] == "mapped"


def test_sample_records_deterministic():
    records = [make_record("nq", i, f"Q{i}?", ["A"]) for i in range(10)]
    a = _sample_records(records, 4, seed=42)
    b = _sample_records(records, 4, seed=42)
    assert [r["source_index"] for r in a] == [r["source_index"] for r in b]


def test_sample_records_n_gte_len_takes_all():
    records = [make_record("nq", i, f"Q{i}?", ["A"]) for i in range(3)]
    out = _sample_records(records, 5, seed=0)
    assert [r["source_index"] for r in out] == [0, 1, 2]


def test_sample_records_sorted_by_source_index():
    records = [make_record("nq", i, f"Q{i}?", ["A"]) for i in range(10)]
    out = _sample_records(records, 3, seed=7)
    idxs = [r["source_index"] for r in out]
    assert idxs == sorted(idxs)
    assert len(idxs) == 3
