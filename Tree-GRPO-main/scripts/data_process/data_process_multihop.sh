#!/usr/bin/env bash
# 多跳 QA 数据预处理：统一走 researchsearch.data.build。
#
# 一次调用同时处理 train / dev 两个划分，跨数据集去重、train/dev 泄漏检测、
# 分配全局唯一整数 index，并输出 train.parquet / dev.parquet / manifest.json。
# 对齐《第 1 周工程闭环开发方案》§4.2，替代旧的 qa_search_train_merge.py /
# qa_search_test_merge.py（旧实现每个数据集的 index 从 0 重置，拼接后 uid 串题）。
#
# 注意：build 输出的是 dev.parquet（开发集），正式 test 集锁定不使用。
# 训练启动时用 DEV_FILE 指向它，例如：export DEV_FILE=$LOCAL_DIR/dev.parquet

set -euo pipefail

WORK_DIR=directory-to-code/TreeGRPO                        # 仓库根目录（Tree-GRPO-main）
LOCAL_DIR=directory-to-data                                # 输出目录
FLASHRAG_DIR=/mnt/workspace/common/datas/FlashRAG_Dataset  # FlashRAG 数据集根目录（目标机）

cd "$WORK_DIR"

python -m researchsearch.data.build \
    --flashrag-dir "$FLASHRAG_DIR" \
    --train-flashrag hotpotqa \
    --dev-flashrag hotpotqa,2wikimultihopqa,musique,bamboogle \
    --out-dir "$LOCAL_DIR" \
    --environment wiki
