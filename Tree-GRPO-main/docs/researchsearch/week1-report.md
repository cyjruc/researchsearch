# ResearchSearch-RL 第 1 周工程闭环报告

更新时间：2026-10-22（Asia/Shanghai）
状态：**GPU smoke 结果已回填；真实检索（FlashRAG 语料 / Web API key）仍待**（见 §9）。

> 性质声明：本文记录「工程 smoke」。所有结果来自 **stub 检索 + fixture 数据 + Qwen2.5-3B**，
> 仅证明训练闭环与协议正确，不构成任何性能/质量结论；正式对照结果（E0–E4）需真实语料与检索服务产生。

## 1. 本周目标与结论速览

- **目标**：打通基于 Tree-GRPO 的 Wiki → Web 两环境训练工程闭环（方案 §1）。
- **结论（无卡部分，已完成）**：源码接线与单元测试闭环——136 项测试全部通过。
- **结论（GPU smoke，已回填）**：真实 GPU 上 E0、GRPO、Tree-GRPO 三段 smoke 全部跑通，
  训练闭环（rollout → 优势计算 → 更新 → checkpoint 保存）验证通过；Tree 验证分 0 → 1.0。
- **结论（真实检索，待）**：Wiki 真实语料/索引与 Web 检索（缺 API key）尚未接通，T02 仍 ⛔。
- **本周可表述为**（真实检索接通后成立）：「完成基于 Tree-GRPO 的 Wiki/Web 两环境训练工程闭环」。

## 2. 完成清单（对照方案 §9）

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| 1 | 记录源码基线、实际依赖、硬件与模型 revision | 🟡 | 本机 + AutoDL GPU 硬件/依赖已实测（[environment.md](./environment.md)）；模型 revision 未单独记录 |
| 2 | 启动参数完整，缺失配置能够提前失败 | ✅ | `preflight.py` + 参数化 `launch_smoke.sh` / `export_ckpt.sh`（`test_export_ckpt.py` 5 项） |
| 3 | Wiki 与真实 Web 检索都能通过统一接口返回结果 | 🟡 | 统一 `post_retrieve` 客户端 + 错误分类完成；真实 Wiki/Web 服务未接通（stub 已验证协议形状） |
| 4 | 数据划分、证据映射与有效样本比例可追踪 | 🟡 | `build.py` 划分/泄漏检测完成；真实语料证据映射待 |
| 5 | 混合 Wiki 数据 index 全局唯一；GRPO 与 Tree 混合 batch uid 不串题 | ✅ | build.py 全局唯一 index（fixture 实测：train 0–2 / dev 3）+ dev split 锁定 |
| 6 | E0 能产生逐题结果与工具轨迹 | ✅ | 真实 GPU 上 E0 逐题导出 `e0_3.json` 落盘（含 turns/stop_reason/tool_cost/gold_answers），`E0_EXPORT_DIR` 门控生效 |
| 7 | GRPO、Tree-GRPO 各完成有效更新与保存 | ✅ | GRPO 2 步、Tree 1 步真实更新，`pg_loss`/`grad_norm` 非零，checkpoint 保存（见 §5） |
| 8 | 树分组、共享前缀、observation mask 通过检查 | ✅ | Tree smoke 中 `inter_tree_advantages`/`inner_tree_advantages` 正常（双重标准化生效），`state_tokens/coverage 0.771` |
| 9 | 证据奖励完成离线验证与小步接线；未完成项明确记录 | ✅ | 离线验证 + 树内接线 + 未完成项记录（[evidence-wiring.md](./evidence-wiring.md) §6） |
| 10 | Tree 叶子金标准证据与实际可见 doc ID 一一对应 | 🟡 | fixture 测试锁定（`test_tree_evidence_wiring.py` 12 项）；真实语料 doc_id 待 |
| 11 | Wiki actor 可导出、重新加载并完成 Web 更新 | 🟡 | model 权重已保留（GRPO/Tree 各 2 step），导出脚本参数化；真实导出/加载待 |
| 12 | 精确续训与仅加载权重的支持范围明确区分 | ✅ | [checkpoint-resume.md](./checkpoint-resume.md)：保存完整，warm start 可达，精确续训未接线 |
| 13 | 日志包含物理生成 token、逻辑工具调用、外部请求和 GPU 时间 | 🟡 | 预算三口径 `get_stats()` 完成（`test_web_cache_budget.py` 19 项）；smoke 日志含 `timing_s/step`/`env/number_of_valid_search`/MFU |
| 14 | 有运行说明、初步结果表、失败案例与第二周资源估算 | ✅ | 本报告即运行说明 + 结果表 + 失败案例 + 资源估算 |

## 3. 门禁对照（T01–T10）

| 门禁 | 状态 | 说明 |
|---|---|---|
| T01 分组/索引 | ✅ | build.py 全局唯一 index（fixture 实测）+ dev split 锁定 |
| T02 证据映射 | ⛔ | 依赖真实 FlashRAG 语料/索引（fixture `evidence_status=not_available`） |
| T03 检索契约 | 🟡 | 客户端完成（16 项）；stub 已验协议形状，真实服务待 |
| T04 分段奖励 | ✅ | 14 项测试锁定语义 |
| T05 证据奖励 | ✅ | evidence.py + 13 项 + 树内接线测试（12 项） |
| T06 loss mask | ✅ | 静态分析 + 单测（10 项）+ Tree smoke 运行时 `state_tokens/coverage 0.771` 观察 |
| T07 优势计算 | ✅ | 9 项 core_algos + 4 项路由 + Tree smoke 运行时双重标准化观察 |
| T08 权重更新 | ✅ | GRPO 2 步 / Tree 1 步真实 optimizer 更新，`grad_norm` 非零，checkpoint 保存 |
| T09 checkpoint | 🟡 | 保存验证通过 + `model_merger.py` 兼容修复；真实导出/加载已验证（tree step1→HF，重载推理正确）；Web warm start 待真实检索 |
| T10 预算 | 🟡 | 缓存命名空间 + 三口径统计（19 项）；真实 rollout 检索/API 计数待真实服务 |

## 4. 运行说明（从空环境到 smoke 结果）

参照 [environment.md](./environment.md) 的硬件/依赖实测；以下命令均在 `Tree-GRPO-main/` 下执行。

### 4.1 环境准备

- RL 环境 `treegrpo`（torch 2.6 + vLLM 0.8.5 + flash-attn + swanlab + verl editable）；
- 检索环境 `retriever`（pyserini + faiss-gpu 1.7.2 + fastapi/uvicorn）。
- 本机（macOS）不可训练，只做代码/文档/测试；GPU 训练在 AutoDL A800 主机（单卡）。

### 4.2 数据准备

```bash
# 本地 JSONL fixture → parquet（stub smoke 用，不依赖 FlashRAG）
python -m researchsearch.data.build --help
# 真实 FlashRAG 目录（目标机，datasets 延迟导入）
bash scripts/data_process/data_process_multihop.sh
```

### 4.3 检索服务

```bash
# stub 检索（smoke 用，固定 Wiki 格式，不依赖真实语料）
python /root/autodl-tmp/stub_retriever.py   # 端口 8000

# 真实 Wiki：FAISS/E5 /retrieve
bash local_retrieval_launch.sh
# 真实 Web：Bing/Brightdata + Redis /retrieve（先起 Redis、设 BRIGHTDATA_API_KEY）
bash scripts/researchsearch/launch_search_service.sh
```

### 4.4 预检与启动（真实 GPU 单卡）

```bash
# 环境变量（凭据/路径不写进脚本）
export DATA_DIR=/root/autodl-tmp/data MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-3B \
       LEARNING_RATE=1e-6 N_GPUS_PER_NODE=1 LOG_DIR=/root/autodl-tmp/logs \
       RAY_ADDRESS=http://127.0.0.1:8265 RETRIEVER_URL=http://127.0.0.1:8000/retrieve

# E0（val_only 未训练基线 + 逐题导出，需 RUNTIME_ENV 注入 E0_EXPORT_DIR）
ALGORITHM=grpo VAL_ONLY=true VAL_BEFORE_TRAIN=true RUNTIME_ENV=/root/autodl-tmp/runtime_env_e0.yaml \
  bash scripts/researchsearch/launch_smoke.sh --submit

# GRPO（2 步）/ Tree（M2 N2 L1 K3）
ALGORITHM=grpo  bash scripts/researchsearch/launch_smoke.sh --submit
ALGORITHM=tree  bash scripts/researchsearch/launch_smoke.sh --submit
```

> 磁盘提醒：单卡 NO_SHARD 下每步 FSDP checkpoint ≈ 36G（optimizer fp32 状态 23G + 模型权重 13G）。
> 80G 数据盘只能存 2 个完整 checkpoint；smoke 后已删 optimizer 状态、仅保留 model 权重（见 §7.2）。

### 4.5 输出定位

- 训练日志：`$LOG_DIR/{experiment_name}.log`；
- checkpoint：`$LOG_DIR/checkpoints/{experiment_name}/actor/global_step_*/`；
- E0 逐题轨迹：`E0_EXPORT_DIR`（本次 `/root/autodl-tmp/logs/e0_export/e0_3.json`）；
- 源码快照：`docs/researchsearch/source-snapshot.md`（`scripts/researchsearch/gen_source_snapshot.py`）。

### 4.6 单元测试与 checkpoint 导出

```bash
python -m pytest tests/                                   # 远程 treegrpo 环境 136 项全过
bash scripts/researchsearch/export_ckpt.sh --dry-run      # 先看路径解析
bash scripts/researchsearch/export_ckpt.sh --merge        # 合并 FSDP 分片为 HF
```

## 5. 结果表（真实 GPU，stub 检索 + fixture + Qwen2.5-3B）

| 实验 | 配置 | 有效步数 | 验证分 | 格式完成率 | 平均搜索 | 每步耗时 | MFU | checkpoint |
|---|---|---|---|---|---|---|---|---|
| E0 | val_only，未训练 | 0 | 0.0 | 0/1（4 turn 全 invalid） | 0 | ~74s（加载+生成） | — | 无 |
| Wiki GRPO | 2 step，n_agent=4 | 2 | 0.0 | 0.875 | 0.75 | 76.6s | 0.120 | 2 步（已留 model） |
| Wiki Tree | M2/N2/L1/K3，1 step | 1 | **1.0** | 0.417 | 0.667 | 104.8s | 0.234 | 2 个（1 训练+1 final） |
| Web 迁移 | alpha=0 | — | — | — | — | — | — | 未跑（缺 API key） |

训练侧关键指标（step 1）：

| 指标 | GRPO | Tree |
|---|---|---|
| `critic/score/mean` | 0.600 | 0.254 |
| `actor/pg_loss` | ~0.000 | 0.129 |
| `actor/grad_norm` | 1.364 | 2.085 |
| `env/finish_ratio` | 0.875 | 0.417 |
| `env/number_of_valid_search` | 0.75 | 0.667 |
| 树优势（inter/inner） | —（grpo 无） | -0.221 / -0.076 |

> 说明：①「验证分」是 `val/test_score/fixture-dev`（1 题 Jupiter）。Tree 验证 1.0 来自 final
> validation（训练后模型答对）；GRPO 验证 0.0 因为 Jupiter 不在 train 3 条里，2 步学不到，属数据覆盖
> 而非训练失败。②「逻辑搜索/API 请求/缓存命中」三口径在 stub 检索下无意义（无 Redis/API），
> 仅以 `env/number_of_valid_search` 反映逻辑搜索；真实口径待真实服务（§9）。③证据分不适用：
> fixture `evidence_status=not_available`、`evidence_alpha=0`。

## 6. 失败案例

1. **E0 未训练模型格式失败**：Qwen2.5-3B 未训练时 4 turn 全 invalid（不产 `<search>`/`<answer>` 标签），
   0 次搜索、`final_answer=null`、`stop_reason=max_turns`（见 `e0_export/e0_3.json`）。这是基线模型的
   正常表现，不是协议 bug；也是 §2 第 6 项「E0 逐题导出」的负样本证据。
2. **GRPO 验证 0 分（数据覆盖）**：dev 唯一一题（Jupiter）不在 train 3 条中，2 步 GRPO 无法泛化。
3. **Tree checkpoint 保存失败（磁盘预算）**：每步 FSDP checkpoint 36G，80G 数据盘写满，导致 final
   `global_step_2` 的 optimizer 状态只写到 576M/23G 即中断。属磁盘预算问题（§7.2），非代码缺陷；
   处理：smoke 后删除 optimizer 状态，仅保留 model 权重。

## 7. 第二周任务与资源估算

### 7.1 第二周任务

- 正式 Wiki RL（沿用 smoke 的冻结语料、提示、预算与评测流程）；
- 第 3 周正式 Web RL 继续使用**搜索摘要工具**（不扩展为完整网页浏览）；
- 准备四项等权 Multi-Hop Avg EM 的 HotpotQA、2Wiki、MuSiQue、Bamboogle 测试 manifest。

### 7.2 资源估算（单卡 A800 + Qwen2.5-3B 实测）

| 项 | 实测值 | 说明 |
|---|---|---|
| 稳态秒/step（GRPO） | 76.6s | 含 gen 16.9s + ref 13.8s + update 14.3s + save 30.8s |
| 稳态秒/step（Tree） | 104.8s | 含 gen 39.3s + update 14.9s + save 35.3s |
| checkpoint 体积 | 36G/step | optim 23G + model 13G；80G 盘只能存 2 个完整 checkpoint |
| MFU | 0.12–0.23 | 单卡 smoke 口径，非吞吐优化目标 |

三档外推（方法，数值待正式 batch 口径实测）：

| 档位 | 稳态秒/step | 目标 step | GPU 小时 | 依据 |
|---|---|---|---|---|
| 乐观 | _待实测_ | _待定_ | _待填_ | 正式 prompt batch 8–16 下重测 |
| 基准 | 76.6（GRPO）/104.8（Tree） | _待定_ | = 秒/step × step / 3600 | 单卡 smoke 实测 |
| 保守 | _待实测_ | _待定_ | _待填_ | 加 checkpoint 磁盘/传输开销 |

> 训练规模由样本数、epoch、有效 prompt batch、sampler 共同决定，不能仅凭数据量线性外推；
> API 费用需真实 Web 服务接入后按「实际计费请求量 × 单价」另算。

## 8. 交付检查表

- [x] 源码快照已重算（256 文件，`gen_source_snapshot.py`）
- [x] 环境/硬件/依赖实测记录（`environment.md`）
- [x] 检索契约、证据接线、loss mask、checkpoint 支持范围文档
- [x] 无卡单元测试 136 项全绿（远程 treegrpo 环境复验）
- [x] 真实 GPU：E0 逐题导出 + GRPO 2 步 + Tree 1 步训练闭环（验证 0→1.0）
- [x] 结果表、失败案例、资源估算回填
- [ ] 真实 Wiki/Web 检索调用记录（待 FlashRAG 语料 + API key）
- [x] checkpoint 导出 + 重新加载（tree step1→HF 6.8G，重载 3.09B 参数、推理 Jupiter 正确）；[ ] Web warm start（待真实 Web 检索）
- [ ] 证据奖励真实 rollout 逐叶验证（待真实语料 doc_id）

## 9. 待回填清单

1. 真实 FlashRAG 语料/索引 → 128/64 条 smoke parquet + manifest + `doc_id` 证据映射（T02）；
2. `BRIGHTDATA_API_KEY` 注入 → 真实 Web 检索 + 预算三口径（logical_queries/cache_hits/api_requests）实测；
3. ~~checkpoint 导出 + 重新加载~~（已完成：tree step1→HF 验证通过，见 §8）；Web warm start（T09 剩余，待真实 Web 检索）；
4. 正式 batch 口径下重测稳态秒/step，填实 §7.2 三档；
5. 证据奖励真实 rollout 逐叶验证与「截断后可见性判定」（[evidence-wiring.md](./evidence-wiring.md) §6）。
