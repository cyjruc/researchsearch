#!/usr/bin/env bash
# ResearchSearch-RL：FSDP checkpoint 导出为 HuggingFace 模型（方案 §6 Day 6 任务 2）。
#
# 复用 scripts/merge_ckpt/model_merger.py，通过本脚本自身目录定位合并器，不依赖
# 当前工作目录。输入输出全部参数化，替换 scripts/merge_ckpt 下硬编码
# /mnt/workspace 路径的旧脚本（convert_checkpoint_from_verl_to_hf.sh，已标注 deprecated）。
#
# 用法（环境变量注入，凭据/路径不写进脚本）：
#   LOCAL_DIR=/path/to/checkpoints/global_step_N/actor \
#   TARGET_DIR=/path/to/merged_hf_model \
#   bash scripts/researchsearch/export_ckpt.sh --merge
#   bash scripts/researchsearch/export_ckpt.sh --dry-run   # 只打印解析后的命令，不执行
#
# FSDP 后端下 verl 会把 HF 模型 config 与 tokenizer 一并写进 checkpoint 目录，因此
# 通常不需要 HF_MODEL_PATH（merger 的 --hf_model_path 已 deprecated）。仅当使用旧
# checkpoint（目录内不含 config）时才需设置 HF_MODEL_PATH。
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MERGER="$(cd -- "${SCRIPT_DIR}/../merge_ckpt" && pwd)/model_merger.py"

ACTION="${1:---merge}"
case "${ACTION}" in
    --merge|--dry-run) ;;
    *)
        printf 'Usage: %s [--merge|--dry-run]\n' "$0" >&2
        exit 2
        ;;
esac

LOCAL_DIR="${LOCAL_DIR:-}"
TARGET_DIR="${TARGET_DIR:-}"
HF_MODEL_PATH="${HF_MODEL_PATH:-}"
BACKEND="${BACKEND:-fsdp}"

if [[ -z "${LOCAL_DIR}" || -z "${TARGET_DIR}" ]]; then
    printf 'LOCAL_DIR and TARGET_DIR are required (export them before calling).\n' >&2
    printf '  LOCAL_DIR: FSDP checkpoint actor dir (contains model_world_size_*_rank_*.pt).\n' >&2
    printf '  TARGET_DIR: directory to write the merged HuggingFace model.\n' >&2
    exit 2
fi
if [[ ! -f "${MERGER}" ]]; then
    printf 'model_merger.py not found at %s\n' "${MERGER}" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    printf 'python3 is required\n' >&2
    exit 1
fi

MERGE_ARGS=(merge --backend "${BACKEND}" --local_dir "${LOCAL_DIR}" --target_dir "${TARGET_DIR}")
if [[ -n "${HF_MODEL_PATH}" ]]; then
    MERGE_ARGS+=(--hf_model_path "${HF_MODEL_PATH}")
fi

if [[ "${ACTION}" == "--dry-run" ]]; then
    printf 'Dry run only. Resolved merge command:\n'
    printf '  python3 "%s"\n' "${MERGER}"
    printf '    %q' "${MERGE_ARGS[@]}"
    printf '\nNo model was merged.\n'
    exit 0
fi

python3 "${MERGER}" "${MERGE_ARGS[@]}"
