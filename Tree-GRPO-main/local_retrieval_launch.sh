#!/bin/bash
# 真实 Wiki 检索服务（FAISS/E5 /retrieve）启动脚本，2 卡适配。
#
# 关键：retrieval_server.py 里 E5 模型 model.cuda() 硬编码、FAISS 用
# faiss.index_cpu_to_all_gpus，会把索引铺到「所有可见 GPU」。2 卡训练已占满双卡，
# 故用 CUDA_VISIBLE_DEVICES 把检索服务固定到单卡（默认卡 1），避免抢卡。
set -Eeuo pipefail

echo "retriever use python:"
which python

INDEX_PATH="${INDEX_PATH:?请设置 INDEX_PATH（如 /root/autodl-tmp/index/e5_Flat.index）}"
CORPUS_PATH="${CORPUS_PATH:?请设置 CORPUS_PATH（如 /root/autodl-tmp/corpus/wiki-18.jsonl）}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:?请设置 RETRIEVER_MODEL_PATH（如 /root/autodl-tmp/models/e5-base-v2）}"
TOPK="${TOPK:-3}"
# 检索服务固定到第几张卡（0 或 1）；E5 向量化在卡上，FAISS 检索默认 CPU。
RETRIEVER_GPU="${RETRIEVER_GPU:-1}"
# FAISS 索引是否放 GPU。2 卡训练与检索共存时默认 CPU（0），避免
# index_cpu_to_all_gpus 把 64GB 索引铺满单卡、与训练抢显存。
FAISS_GPU="${FAISS_GPU:-0}"

export CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU}"

EXTRA_ARGS=()
if [[ "${FAISS_GPU}" == "1" ]]; then
    EXTRA_ARGS+=(--faiss_gpu)
fi

python search_r1/search/retrieval_server.py --index_path "$INDEX_PATH" \
                                            --corpus_path "$CORPUS_PATH" \
                                            --topk "$TOPK" \
                                            --retriever_name "$RETRIEVER_NAME" \
                                            --retriever_model "$RETRIEVER_MODEL_PATH" \
                                            "${EXTRA_ARGS[@]}"
