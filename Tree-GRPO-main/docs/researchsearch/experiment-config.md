# Wiki 阶段实验配置统一（E1 / E2 / E3）

版本：v1.0 · 2026-10-22
对应脚本：`scripts/researchsearch/launch_wiki.sh`
对应方案：《第 2 周 Wiki 实验开发方案》§5、门禁 E03；架构 §3「E1/E2 公平对照前置条件」。

## 1. 目标与原则

三个 Wiki 实验共享起始模型、数据划分、工具、提示词与评测设置（架构 §3）：

| 实验 | 入口 | 方法 | 奖励 |
|---|---|---|---|
| E1 | `verl.trainer.main_ppo_format` | GRPO | 答案奖励 |
| E2 | `verl.trainer.main_ppo_format_ts` | Tree-GRPO | 答案奖励 |
| E3 | `verl.trainer.main_ppo_format_ts` | Tree-GRPO | 答案＋证据奖励 |

「答案奖励」均指相同配置的 `R_existing`（含现有格式分段评分），且 `retrieval_score=0`。仅 E3 启用独立证据项 `evidence_alpha > 0`。

公平对照原则：**可统一的学习率、优化器、warmup、有效 prompt batch、KL 设置、生成长度及数据顺序，在脚本里集中定义、三实验一致；算法相关差异显式分叉并记录，不强行抹平。**

## 2. 统一项清单

「冻结」= 架构硬约束，不再改；「待校准」= 起步值，Day 4 小规模检查后冻结。

| 项 | 配置键 | 统一默认 | 状态 |
|---|---|---|---|
| 模型 | `actor_rollout_ref.model.path` / `MODEL_PATH` | Qwen2.5-3B | 冻结（base/instruct 与 revision 须明确，`MODEL_REVISION` 待固定） |
| 学习率 | `actor_rollout_ref.actor.optim.lr` | 1e-6 | 待校准 |
| warmup | `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | 0.0 | 待校准 |
| 有效 prompt batch | `data.train_batch_size` | 16 | 待校准（§5.1 起步 8–16） |
| val batch | `data.val_batch_size` | 16 | 待校准 |
| mini / micro batch | `actor_rollout_ref.actor.ppo_mini/micro_batch_size` | 32 / 8 | 待校准 |
| KL | `use_kl_loss` / `kl_loss_coef` / `kl_loss_type` | true / 0.001 / low_var_kl | 冻结 |
| 生成长度 | `data.max_prompt/response/start/obs_length` | 1024 / 256 / 512 / 256 | 待校准（`max_response_length` 是每轮上限，非整条轨迹） |
| 数据顺序 | 见「待确认项」 | — | 待确认（seed 注入点未定位） |
| 奖励分段 | `reward_model.structure/final_format_score` | 0.2 / 0.1 | 冻结 |
| 检索奖励 | `reward_model.retrieval_score` | 0 | 冻结（本项目固定为 0） |
| 工具 | `do_search` / `retriever.url` / `retriever.topk` | true / 检索服务 / 3 | 冻结（训练预算；评测预算另定且 ≥10） |
| 温度 | `actor_rollout_ref.rollout.temperature` | 1.0 | 冻结 |
| 优化器 / FSDP offload | `fsdp_config.param/grad/optimizer_offload` | true / true / true | 冻结（4-GPU 正式口径） |
| GPU 显存占用 | `rollout.gpu_memory_utilization` | 0.6 | 冻结（4-GPU 正式口径） |

## 3. 算法相关差异（唯一分叉点）

| 参数 | E1 | E2 | E3 |
|---|---|---|---|
| `algorithm.adv_estimator` | `grpo` | `tree` | `tree` |
| 入口模块 | `main_ppo_format` | `main_ppo_format_ts` | `main_ppo_format_ts` |
| 同题采样数 `n_agent` | 4 | 1 | 1 |
| 树搜索 `tree_search` | — | true | true |
| 树形 `ts_m / ts_n / ts_l / ts_k` | — | 2 / 2 / 1 / 3 | 2 / 2 / 1 / 3 |
| 证据权重 `reward_model.evidence_alpha` | —（默认 0） | —（默认 0） | `EVIDENCE_ALPHA`（默认 0.1 验证值） |

语义注意（架构 §3 / CLAUDE.md）：

- `n_agent` 是普通 GRPO 的同题采样数，实际轨迹条数 = `n_agent × train_batch_size`。
- `ts_k`（K）是每树最终抽样叶子数，**不等于独立轨迹数或物理生成成本**；Tree 的最终叶子槽位为每题 `ts_m × ts_k`。
- `ts_m / ts_n / ts_l / ts_k` 四个值必须与启动脚本一致并写入运行记录，不得只记 M/N/L/K 而省略配置键。

## 4. 预算停止规则（预先固定）

| 项 | 配置键 | 状态 |
|---|---|---|
| 总训练步数 | `trainer.total_training_steps` | 待 Day 4 冻结（提交前必须非空） |
| epoch | `trainer.total_epochs` | 待 Day 4 冻结 |
| 保存 / 评测频率 | `trainer.save_freq` / `trainer.test_freq` | 待 Day 4 冻结 |

成本口径（架构 §3）：分别报告物理新生成 token、逻辑搜索次数、实际外部请求（含重试）、GPU 时间；共享前缀不重复计入物理新生成 token。轨迹条数相同不代表成本相同。

## 5. 证据权重命名与选值流程

统一使用代码配置名 `evidence_alpha`（即架构 §2.1 的 `evidence_weight` / α）。选值流程：

```
ppo_trainer.yaml 的 reward_model.evidence_alpha（默认 0.0）
  → main_ppo_format_ts.py:209  config.reward_model.get('evidence_alpha', 0.0)
  → RewardManager.__init__(evidence_alpha=...)
```

- 默认 `0.0` 关闭证据项（E1/E2 不 override，走 yaml 默认）。
- `0.1` 仅作接线验证值，不视为最终超参。
- E3 正式 `evidence_alpha` 由第 2 周 §7 消融确定（E3 vs E2）。
- 无可靠 gold 时证据项关闭（`eligible=False`）并记录覆盖范围。

## 6. 待确认 / 待校准项（Day 4 前收口）

1. **数据顺序 seed 注入点**：verl 0.8.5 的 FSDP `ppo_trainer.yaml` 未见顶层 `seed` 配置键（仅 megatron / sft 配置含 seed）。需在 Day 4 前确认 trainer 是否通过 `set_seed` / 环境变量 / dataloader 层控制数据顺序；若无法统一，则如实记录为「无法统一的算法相关参数」，不伪称已统一。
2. **`MODEL_REVISION`**：固定到 Qwen2.5-3B 的具体 revision（base/instruct 二选一），三个实验一致。
3. **batch / 生成长度 / 步数 / 保存频率**：按 Day 4 小规模检查（128/64 或 1K 子集，10–20 步）实测校准后冻结，回填脚本默认值。
4. **`TOTAL_TRAINING_STEPS`**：由样本数、epoch、有效 prompt batch 与 sampler 共同决定，提交前必须显式非空。

## 7. 与第 1 周 smoke 的分界

- smoke 用单卡内存策略（`param/grad_offload=false`、`optimizer_offload=true`、`gpu_memory_utilization=0.3`）；正式 4-GPU 用 `param/grad/optimizer_offload=true`、`0.6`。二者不可混用。
- smoke 的 `val_only` / `val_before_train` / `PROBE_RETRIEVER` 等 E0 协议开关仅存在于 `launch_smoke.sh`，正式脚本不含。
- 本脚本不覆盖评测脚本；评测预算（Recall@10 须 ≥10 候选）与训练预算分开固定（方案 §3.3）。

## 8. 验收（门禁 E03）

- 可统一超参三实验逐项一致（由 `COMMON_OVERRIDES` 单一来源保证）。
- 算法相关差异已记录（§3 表），预算停止规则固定（§4）。
- 本地 `--dry-run` 可验证：`e1/e2/e3` 打印出的命令只在「算法相关差异」处不同。
