#!/usr/bin/env bash
# Deprecated：本脚本硬编码 /mnt/workspace 作者机器路径，仅作历史参考。
# 请改用参数化包装脚本 scripts/researchsearch/export_ckpt.sh（方案 §6 Day 6 任务 2），
# 它通过自身目录定位 model_merger.py 并参数化输入输出：
#
#   LOCAL_DIR=/path/to/checkpoints/global_step_N/actor \
#   TARGET_DIR=/path/to/merged_hf_model \
#   bash scripts/researchsearch/export_ckpt.sh --merge
#
# 原硬编码调用保留在下方注释，便于对照参数含义。
#
#   python ./model_merger.py merge \
#       --backend fsdp \
#       --hf_model_path /mnt/workspace/common/models/Qwen2.5-3B \
#       --local_dir /mnt/workspace/TreeGRPO/verl_checkpoints/.../actor/global_step_160 \
#       --target_dir /mnt/workspace/TreeGRPO/scripts/merge_ckpt/qwen2.5-ts
exit 1
