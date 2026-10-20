#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PREFLIGHT="${SCRIPT_DIR}/preflight.py"

ACTION="${1:---dry-run}"
case "${ACTION}" in
    --dry-run|--submit) ;;
    *)
        printf 'Usage: %s [--dry-run|--submit]\n' "$0" >&2
        exit 2
        ;;
esac

DATA_DIR="${DATA_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-}"
LOG_DIR="${LOG_DIR:-}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
ALGORITHM="${ALGORITHM:-grpo}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
DEV_FILE="${DEV_FILE:-${DATA_DIR}/dev.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-day1-${ALGORITHM}-smoke}"
SAVE_DIR="${SAVE_DIR:-${LOG_DIR}/checkpoints/${EXPERIMENT_NAME}}"
# 起步配置说明：以下 batch 与 tree 参数是 2-step smoke 的缩小值，不是正式 E1/E2 口径。
# 方案 §5.1：正式 prompt batch 8–16；ts_k=3；Wiki 预算最大 3 次搜索、topk=3。
# PPO_MINI_BATCH_SIZE 按 repeat 后的序列数理解：GRPO = n_agent×batch，Tree = M×K×batch。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-2}"
TREE_M="${TREE_M:-2}"
TREE_N="${TREE_N:-2}"
TREE_L="${TREE_L:-1}"
TREE_K="${TREE_K:-3}"
MAX_TURNS="${MAX_TURNS:-3}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"

export DATA_DIR MODEL_PATH LEARNING_RATE N_GPUS_PER_NODE LOG_DIR RAY_ADDRESS RETRIEVER_URL TRAIN_FILE DEV_FILE

if [[ "${ALGORITHM}" != "grpo" && "${ALGORITHM}" != "tree" ]]; then
    printf 'ALGORITHM must be grpo or tree\n' >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    printf 'python3 is required\n' >&2
    exit 1
fi

python3 "${PREFLIGHT}" --for-launch
mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

if [[ "${ALGORITHM}" == "grpo" ]]; then
    MAIN_MODULE="verl.trainer.main_ppo_format"
else
    MAIN_MODULE="verl.trainer.main_ppo_format_ts"
fi

COMMON_OVERRIDES=(
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${DEV_FILE}"
    "data.train_data_num=null"
    "data.val_data_num=null"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.max_prompt_length=1024"
    "data.max_response_length=256"
    "data.max_start_length=512"
    "data.max_obs_length=256"
    "data.shuffle_train_dataloader=true"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.actor.policy_loss=grpo"
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0"
    "actor_rollout_ref.actor.use_kl_loss=true"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=true"
    "actor_rollout_ref.actor.fsdp_config.grad_offload=true"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size=${PPO_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
    "actor_rollout_ref.ref.log_prob_micro_batch_size=${PPO_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=true"
    "actor_rollout_ref.actor.kl_loss_coef=0.001"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "algorithm.no_think_rl=false"
    "algorithm.use_kl_in_reward=false"
    "actor_rollout_ref.rollout.temperature=1"
    "actor_rollout_ref.actor.state_masking=true"
    "trainer.logger=['console']"
    "+trainer.val_only=false"
    "+trainer.val_before_train=false"
    "trainer.default_hdfs_dir=null"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=1"
    "trainer.save_freq=1"
    "trainer.test_freq=0"
    "trainer.project_name=ResearchSearch-RL"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=${MAX_TRAINING_STEPS}"
    "trainer.default_local_dir=${SAVE_DIR}"
    "reward_model.structure_format_score=0.2"
    "reward_model.final_format_score=0.1"
    "reward_model.retrieval_score=0"
    "do_search=true"
    "max_turns=${MAX_TURNS}"
    "retriever.url=${RETRIEVER_URL}"
    "retriever.topk=${RETRIEVER_TOPK}"
)

if [[ "${ALGORITHM}" == "grpo" ]]; then
    ALGORITHM_OVERRIDES=(
        "algorithm.adv_estimator=grpo"
        "actor_rollout_ref.rollout.n_agent=4"
    )
else
    ALGORITHM_OVERRIDES=(
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
fi

JOB_COMMAND=(
    ray job submit
    "--address=${RAY_ADDRESS}"
    "--runtime-env=verl/trainer/runtime_env.yaml"
    --
    python3 -m "${MAIN_MODULE}"
    "${COMMON_OVERRIDES[@]}"
    "${ALGORITHM_OVERRIDES[@]}"
)

cd "${REPO_ROOT}"
if [[ "${ACTION}" == "--dry-run" ]]; then
    printf 'Dry run only. Resolved launch command:\n'
    printf '  %q' "${JOB_COMMAND[@]}"
    printf '\nNo Ray job was submitted.\n'
    exit 0
fi

python3 "${PREFLIGHT}" --check-runtime --check-ray --check-ray-worker
if [[ "${PROBE_RETRIEVER:-0}" == "1" ]]; then
    python3 "${PREFLIGHT}" --probe-retriever
fi

RUN_LOG="${LOG_DIR}/${EXPERIMENT_NAME}.log"
printf 'Submitting %s smoke run; console log: %s\n' "${ALGORITHM}" "${RUN_LOG}"
"${JOB_COMMAND[@]}" 2>&1 | tee "${RUN_LOG}"
