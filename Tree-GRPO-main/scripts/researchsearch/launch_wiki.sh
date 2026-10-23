#!/usr/bin/env bash
# Wiki 阶段正式训练统一启动脚本（E1 / E2 / E3）。
#
# 目标：把架构 §3 与第二周方案 §5 要求的「E1/E2 公平对照」落实为单一脚本——
# 三实验共享所有可统一超参，只在「算法相关差异」上分叉，预算停止规则预先固定。
#
# 用法：
#   ./launch_wiki.sh <e1|e2|e3> [--dry-run|--submit]
#   ./launch_wiki.sh e1           # 默认 --dry-run：本地（无卡）解析并打印最终 ray 命令
#   ./launch_wiki.sh e3 --submit  # 提交前跑 preflight，再提交 Ray 作业
#
# 实验对照（架构 §3）：
#   E1 = GRPO，答案奖励           （普通 GRPO 基线）
#   E2 = Tree-GRPO，答案奖励      （验证树式方法）
#   E3 = Tree-GRPO，答案＋证据奖励（完整项目方案，仅 E3 启用 evidence_alpha > 0）
#
# 所有「答案奖励」均指相同配置的 R_existing（含现有格式分段评分），retrieval_score=0。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PREFLIGHT="${SCRIPT_DIR}/preflight.py"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
EXPERIMENT="${1:-}"
ACTION="${2:---dry-run}"

case "${EXPERIMENT}" in
    e1|e2|e3) ;;
    *)
        printf 'Usage: %s <e1|e2|e3> [--dry-run|--submit]\n' "$0" >&2
        exit 2
        ;;
esac
case "${ACTION}" in
    --dry-run|--submit) ;;
    *)
        printf 'Usage: %s <e1|e2|e3> [--dry-run|--submit]\n' "$0" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# 一、可统一超参（三实验必须一致；「冻结」为架构硬约束，「待校准」为 Day 4 定值）
# ---------------------------------------------------------------------------
# 模型（架构 §5.1：固定 Qwen2.5-3B，base/instruct 与 revision 明确，不混用）
MODEL_PATH="${MODEL_PATH:-}"
MODEL_REVISION="${MODEL_REVISION:-}"   # 待冻结：固定到具体 revision 后写入

# 数据（12K Wiki train + 独立 dev；TRAIN_FILE/DEV_FILE 由 DATA_DIR 推导）
DATA_DIR="${DATA_DIR:-}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
DEV_FILE="${DEV_FILE:-${DATA_DIR}/dev.parquet}"

# 优化器 / 学习率（架构 §5.1：起步 1e-6，正式值由 Day 4 小规模检查校准）
LEARNING_RATE="${LEARNING_RATE:-1e-6}"          # 待校准
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.0}"  # 待校准

# batch（架构 §5.1：prompt batch 起步 8–16；12K 正式按 GPU 与吞吐校准）
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"      # 待校准
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"          # 待校准
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"  # 待校准
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"  # 待校准
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"  # 待校准

# KL（三实验统一）
USE_KL_LOSS="${USE_KL_LOSS:-true}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"

# 生成长度（统一；注意 data.max_response_length 是「每轮」上限，非整条轨迹预算）
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"    # 待校准
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}" # 待校准
MAX_START_LENGTH="${MAX_START_LENGTH:-512}"       # 待校准
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-256}"           # 待校准

# 数据顺序（架构要求「同 seed 数据顺序」；见文档「待确认项」：verl 0.8.5 FSDP
# ppo 路径未见顶层 seed 配置键，需 Day 4 前确认注入点，否则如实记为无法统一）
DATA_SEED="${DATA_SEED:-}"

# 奖励分段（架构 §2.1：retrieval_score=0 为固定值，格式分段三实验一致）
STRUCTURE_FORMAT_SCORE="${STRUCTURE_FORMAT_SCORE:-0.2}"
FINAL_FORMAT_SCORE="${FINAL_FORMAT_SCORE:-0.1}"
RETRIEVAL_SCORE="${RETRIEVAL_SCORE:-0}"   # 冻结：本项目固定为 0

# 硬件相关（4-GPU 正式口径；单卡 smoke 的内存策略见 launch_smoke.sh，不可混用）
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-true}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-true}"

# ---------------------------------------------------------------------------
# 二、预算停止规则（架构 §3：预先固定）
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"   # 待 Day 4 冻结（12K 正式步数）
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"                  # 待 Day 4 冻结
SAVE_FREQ="${SAVE_FREQ:-100}"                      # 待 Day 4 冻结
TEST_FREQ="${TEST_FREQ:-100}"                      # 待 Day 4 冻结（评测频率）

# ---------------------------------------------------------------------------
# 三、检索 / 工具预算（训练预算；评测预算在评测脚本单独固定且 ≥10，见方案 §3.3）
# ---------------------------------------------------------------------------
MAX_TURNS="${MAX_TURNS:-3}"        # 训练期搜索轮次上限
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"  # 每轮返回文档数
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

# ---------------------------------------------------------------------------
# 四、运行与日志
# ---------------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
LOGGER="${LOGGER:-['console']}"     # 正式可设 ['swanlab','console']，key 经环境变量注入
RUNTIME_ENV="${RUNTIME_ENV:-verl/trainer/runtime_env.yaml}"

case "${EXPERIMENT}" in
    e1) EXPERIMENT_NAME="wiki-e1-grpo-qwen2.5-3b" ;;
    e2) EXPERIMENT_NAME="wiki-e2-tree-qwen2.5-3b" ;;
    e3) EXPERIMENT_NAME="wiki-e3-tree-evidence-qwen2.5-3b" ;;
esac
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOG_DIR}/checkpoints/${EXPERIMENT_NAME}}"

# ---------------------------------------------------------------------------
# 五、算法相关差异（架构 §3：记录，不强行抹平）
#     E1 记录同题采样数 n_agent；E2/E3 完整记录 M/N/L/K（ts_m/ts_n/ts_l/ts_k）。
#     注意：K 是每树最终抽样叶子数，不能等同于独立轨迹数或物理生成成本。
# ---------------------------------------------------------------------------
N_AGENT="${N_AGENT:-4}"   # E1 同题采样数：实际轨迹条数 = n_agent × train_batch_size
TREE_M="${TREE_M:-2}"     # ts_m
TREE_N="${TREE_N:-2}"     # ts_n
TREE_L="${TREE_L:-1}"     # ts_l
TREE_K="${TREE_K:-3}"     # ts_k：每树最终抽样叶子数（第一周 §5.1 起步值）
# 证据奖励权重（仅 E3 启用；架构 §2.1：默认 0 关闭，0.1 仅接线验证值，正式由 §7 消融确定）
EVIDENCE_ALPHA="${EVIDENCE_ALPHA:-0.1}"

case "${EXPERIMENT}" in
    e1)
        MAIN_MODULE="verl.trainer.main_ppo_format"
        EXPERIMENT_OVERRIDES=(
            "algorithm.adv_estimator=grpo"
            "actor_rollout_ref.rollout.n_agent=${N_AGENT}"
        )
        ;;
    e2)
        MAIN_MODULE="verl.trainer.main_ppo_format_ts"
        EXPERIMENT_OVERRIDES=(
            "algorithm.adv_estimator=tree"
            "actor_rollout_ref.rollout.tree_search=true"
            "actor_rollout_ref.rollout.ts_m=${TREE_M}"
            "actor_rollout_ref.rollout.ts_n=${TREE_N}"
            "actor_rollout_ref.rollout.ts_l=${TREE_L}"
            "actor_rollout_ref.rollout.ts_k=${TREE_K}"
            "actor_rollout_ref.rollout.reward_mode=base"
            "actor_rollout_ref.rollout.expand_mode=random"
            "actor_rollout_ref.rollout.n_agent=1"
        )
        ;;
    e3)
        MAIN_MODULE="verl.trainer.main_ppo_format_ts"
        EXPERIMENT_OVERRIDES=(
            "algorithm.adv_estimator=tree"
            "actor_rollout_ref.rollout.tree_search=true"
            "actor_rollout_ref.rollout.ts_m=${TREE_M}"
            "actor_rollout_ref.rollout.ts_n=${TREE_N}"
            "actor_rollout_ref.rollout.ts_l=${TREE_L}"
            "actor_rollout_ref.rollout.ts_k=${TREE_K}"
            "actor_rollout_ref.rollout.reward_mode=base"
            "actor_rollout_ref.rollout.expand_mode=random"
            "actor_rollout_ref.rollout.n_agent=1"
            "reward_model.evidence_alpha=${EVIDENCE_ALPHA}"
        )
        ;;
esac

# ---------------------------------------------------------------------------
# 六、共享 override（三实验逐项一致）
# ---------------------------------------------------------------------------
COMMON_OVERRIDES=(
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${DEV_FILE}"
    "data.train_data_num=null"
    "data.val_data_num=null"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.max_start_length=${MAX_START_LENGTH}"
    "data.max_obs_length=${MAX_OBS_LENGTH}"
    "data.shuffle_train_dataloader=true"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.actor.policy_loss=grpo"
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO}"
    "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.grad_offload=${GRAD_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size=${LOG_PROB_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "+actor_rollout_ref.rollout.disable_log_stats=true"
    "+actor_rollout_ref.rollout.enable_chunked_prefill=true"
    "actor_rollout_ref.ref.log_prob_micro_batch_size=${LOG_PROB_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "algorithm.no_think_rl=false"
    "algorithm.use_kl_in_reward=false"
    "actor_rollout_ref.rollout.temperature=1"
    "actor_rollout_ref.actor.state_masking=true"
    "trainer.logger=${LOGGER}"
    "trainer.default_hdfs_dir=null"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.project_name=ResearchSearch-RL"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.default_local_dir=${CHECKPOINT_DIR}"
    "reward_model.structure_format_score=${STRUCTURE_FORMAT_SCORE}"
    "reward_model.final_format_score=${FINAL_FORMAT_SCORE}"
    "reward_model.retrieval_score=${RETRIEVAL_SCORE}"
    "do_search=true"
    "max_turns=${MAX_TURNS}"
    "retriever.url=${RETRIEVER_URL}"
    "retriever.topk=${RETRIEVER_TOPK}"
)

JOB_COMMAND=(
    ray job submit
    "--address=${RAY_ADDRESS}"
    "--runtime-env=${RUNTIME_ENV}"
    --
    python3 -m "${MAIN_MODULE}"
    "${COMMON_OVERRIDES[@]}"
    "${EXPERIMENT_OVERRIDES[@]}"
)

# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
if [[ "${ACTION}" == "--dry-run" ]]; then
    printf '== 实验 %s（%s）dry-run，统一后的完整启动命令 ==\n' "${EXPERIMENT}" "${EXPERIMENT_NAME}"
    printf '  %q' "${JOB_COMMAND[@]}"
    printf '\n\n-- 算法相关差异（唯一的分叉点）--\n'
    printf '  %s\n' "${EXPERIMENT_OVERRIDES[@]}"
    printf '\n-- 提示 --\n'
    printf '  其余 COMMON_OVERRIDES（%d 项）三实验逐项一致；未提交任何 Ray 作业。\n' "${#COMMON_OVERRIDES[@]}"
    printf '  待校准项（Day 4 冻结）：LEARNING_RATE、batch、生成长度、TOTAL_TRAINING_STEPS/EPOCHS、SAVE/TEST_FREQ。\n'
    if [[ -z "${TOTAL_TRAINING_STEPS}" ]]; then
        printf '  警告：TOTAL_TRAINING_STEPS 为空，提交前必须由 Day 4 冻结。\n'
    fi
    exit 0
fi

# --submit：先本地预检（缺数据/缺模型/缺学习率在提交前失败），再提交
export DATA_DIR MODEL_PATH LEARNING_RATE N_GPUS_PER_NODE LOG_DIR RAY_ADDRESS RETRIEVER_URL
python3 "${PREFLIGHT}" --for-launch
python3 "${PREFLIGHT}" --check-runtime --check-ray --check-ray-worker

RUN_LOG="${LOG_DIR}/${EXPERIMENT_NAME}.log"
printf 'Submitting %s（%s）; console log: %s\n' "${EXPERIMENT}" "${EXPERIMENT_NAME}" "${RUN_LOG}"
cd "${REPO_ROOT}"
"${JOB_COMMAND[@]}" 2>&1 | tee "${RUN_LOG}"
