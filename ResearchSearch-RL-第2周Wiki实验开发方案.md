# ResearchSearch-RL：第 2 周 Wiki 实验开发方案

版本：v1.0 · 2026-10-22
项目根目录：`/Users/ct/code/算法/researchsearch`
代码目录：`/Users/ct/code/算法/researchsearch/Tree-GRPO-main`
文档位置：项目根目录 `ResearchSearch-RL-第2周Wiki实验开发方案.md`。下文源码路径均相对于代码目录。

本方案承接[第 1 周工程闭环开发方案](./ResearchSearch-RL-第1周工程闭环开发方案.md)，目标按[项目架构与实验设计](./ResearchSearch-RL-项目架构与实验设计.md) §5「第 2 周：Wiki 实验」执行。第 1 周已完成工程闭环（E0/GRPO/Tree 三段 smoke 跑通、T06/T08/T09 门禁转 ✅、树结构与 E0 逐题导出、checkpoint 导出重载验证），本周把焦点从「链路能跑」转到「Wiki 阶段的三个正式实验与证据奖励」：完成 12K 数据与证据映射、打通真实 doc_id 并验证树内证据奖励（接线代码已在第 1 周 Tier2 落地）、先小规模检查再运行 E1/E2/E3 的 Wiki 阶段、得到三个 Wiki checkpoint、完成 EM/Recall@K 与奖励消融。

## 1. 本周目标与边界

在已跑通的工程闭环之上，完成 Wiki 阶段可复核的正式实验：

```text
冻结 Wiki 语料 + FAISS 索引
  → 12K 训练数据 + 证据映射（gold → 稳定 doc_id）
  → 树内证据奖励接线验证（代码已在第 1 周落地，打通真实 doc_id）
  → 统一 E1/E2/E3 配置 → 小规模检查 → 12K 正式训练
  → 三个 Wiki checkpoint（E1 GRPO、E2 Tree、E3 Tree+证据）
  → EM / Recall@K 评测 + 奖励消融（E2 vs E1、E3 vs E2）
```

本周成功标准是**三个 Wiki checkpoint 可复核、消融结论可追溯到逐题结果**，不要求复现上游 31.8 或简历中的 36.2 等数值。Web 迁移（第三周）、2.2K Web 数据、四项 Web benchmark 全量评测、多随机种子与预算对照（第四周）不在本周范围。本周不默认做 SFT、不训练检索器、不增加 BGE 对照。

三条不变约束沿用第 1 周：

- 工具返回内容（`<information>`）不作为模型生成 token 参与策略损失（observation mask 已在 smoke 验证）。
- 证据字段只用于奖励与评测，不进入模型 prompt 或工具观察。
- 预期数值是待验证假设，不是验收门槛；未提升的指标如实报告。

## 2. 第 1 周承接：已完成与遗留

第 1 周结束时，门禁状态与本周动作如下。

| 门禁 | 第 1 周末状态 | 本周动作 |
|---|---|---|
| T01 分组/索引 | ✅ | 12K 数据沿用 build.py 全局唯一 `index`，无新工作 |
| T02 证据元数据 | ⛔ | 接线代码已在 Tier2 落地；本周完成「数据证据映射」（§3）与「真实 doc_id 打通 + 接线验证」（§4） |
| T03 检索契约 | 🟡 | 真实索引就绪后补 `/retrieve` 契约测试与 doc_id 回查（§3） |
| T04 分段奖励 | ✅ | 沿用，不重写 |
| T05 证据奖励 | 🟡 | 纯函数 + 树内接线已在 Tier2 落地；本周打通真实 doc_id 与 alpha 消融（§3、§4、§7） |
| T06 loss mask | ✅ | 已通过真实 smoke，无新工作 |
| T07 优势计算 | ✅ | 沿用 `adv_estimator=tree` 双归一化路由，无新工作 |
| T08 权重更新 | ✅ | 已通过真实 smoke，无新工作 |
| T09 checkpoint | ✅ | 复用已修复的 model_merger 导出（§6） |
| T10 预算 | ⛔/🟡 | 本周正式训练补真实生成 token / 逻辑搜索 / 外部请求 / GPU 时间统计（§6） |

第 1 周已交付、本周直接复用的资产：

- 数据层：`researchsearch/data/build.py`（全局唯一 `index`、规范化去重、train/dev 泄漏检测、`evidence_status` 四态、manifest、`load_flashrag_source(dir, source, split, evidence_map_fn=...)`）。
- 证据纯函数：`verl/utils/reward_score/evidence.py`（`compute_evidence_reward`、`extract_doc_ids_from_retrieval`、`compute_recall_at_k`、`_dedup_preserve_order`）。
- 接线蓝图：`docs/researchsearch/evidence-wiring.md`（三个丢失点 + 叶子顺序对齐不变量 + 三处最小改动方案）。
- 导出：`scripts/researchsearch/export_ckpt.sh` + 已兼容 verl 0.8.5 的 `scripts/merge_ckpt/model_merger.py`。
- 启动/预检：`scripts/researchsearch/launch_smoke.sh`、`preflight.py`、`ray_worker_probe.py`。

遗留阻塞项（本周能否解除取决于资源是否到位，须如实标注）：

- **真实 FlashRAG 语料/索引**（T02 数据部分）：尚未提供，本周 Day 1–2 的首要外部依赖。
- **真实 doc_id 打通**（T02 接线部分收尾）：证据接线代码已在 Tier2 落地（`extra_info` 进 `gen_batch` → 逐叶 `visible_doc_ids` → `RewardManager` 读 `evidence_alpha`），但 Wiki retriever 尚未返回稳定 `document.id`，真实覆盖率仍为空；随真实语料打通（§3、§4）。
- **精确续训支持范围调查**（第 1 周 Day 7 遗留）：本周不扩展，仍以「warm start + 新建优化器」为两阶段衔接口径；若本周有富余再补调查报告。

## 3. 数据就绪：12K Wiki 数据与证据映射

### 3.1 规模与划分

沿用架构 §2 的 Wiki 训练规模，本周生成正式数据：

| 数据 | 规模 | 用途 |
|---|---:|---|
| Wiki train | NQ 2K + HotpotQA 4K + 2Wiki 3K + MuSiQue 3K = 12K | E1/E2/E3 正式训练 |
| Wiki dev | 建议 512–1024 条，独立划分 | 阶段评测与早停观察；不与 train、smoke dev 重叠 |
| 锁定测试集 | HotpotQA、2Wiki、MuSiQue、Bamboogle | 正式 Multi-Hop Avg EM；本周只预注册 manifest，不训练 |

数据隔离规则沿用第 1 周：源数据 train/dev/test 不混用，**正式测试集锁定**，本周训练与阶段评测只用 train/dev。`build.py` 的 `--dev-flashrag` 已改为取官方 `dev` split（不消费 `test`）。**数据加载器 `load_flashrag_source` 在指定 split 不存在时直接报错，不自动回退到 dev/train**——回退会让 dev 悄悄落到 train、破坏隔离，正式生成必须显式指定存在的 split。合成 fixture 不得混入真实评测。

### 3.2 证据映射（T02 数据部分，本周核心外部依赖）

目标：把 FlashRAG 各数据集的结构化支持事实映射到冻结 Wikipedia 语料的稳定 doc_id，写入 `extra_info.supporting_doc_ids` 与 `evidence_status`。

技术路径：

1. 固定语料与版本：确定 Wikipedia dump 版本、切块方式与标题→doc_id 映射表，`corpus_version` 固定；同一冻结语料下重复查询可复核。
2. 建立 E5 + FAISS 索引，核验向量维度、语料条数与索引条目映射（第 1 周 Day 2 未完成项）。
3. 实现 `evidence_map_fn(example, idx)`（`build.py::load_flashrag_source` 已预留该钩子）：把 FlashRAG 的 supporting facts（`[title, sentence_index]` 对）按 title 定位到语料文档，聚合为 `supporting_doc_ids`。
4. `evidence_status` 判定：全部支持文档可映射 → `mapped`；部分可映射 → `partial`；无支持事实 → `not_available`；映射关系歧义（同名 title 多文档）→ 人工核查后定 `mapped`/`partial`/`unmapped`。
5. 冻结语料一致性检查（架构 §2.1 明确要求）：核对金标准文档是否存在于索引；语料缺失误判为搜索策略失败，须单独统计并报告。

交付：

- `train.parquet` / `dev.parquet`（12K + dev）与 `manifest.json`（含抽样种子、样本清单、`index ↔ sample_id` 一一对应）。
- 证据映射小样本报告：可映射率、partial 比例、歧义案例、语料缺失案例；至少 20 条人工核对。
- `docs/researchsearch/retrieval-contract.md` 补真实 doc_id 回查结论。

验收（对应第 1 周 T02 数据部分）：混合数据 `index` 无碰撞且每个 `index` 只映射一条 `sample_id`；train/dev ID 无交集；`supporting_doc_ids` 可回查正文；证据字段不进入 prompt。

### 3.3 检索预算与 Recall@K 前置

架构 §4 明确：正式 Recall@10 前，检索预算必须允许返回至少 10 个不同文档。第 1 周 smoke 的三轮 top-3 最多 9 个候选，仅用于计算逻辑验证。本周正式训练的检索预算需在评测配置中统一增大（例如 4 轮 top-3 = 12 候选，或 5 轮 top-3 = 15 候选，去重后如实统计真实候选数，不补齐、不选择性剔除）。训练预算与评测预算分开固定，评测预算须 ≥10。

## 4. 证据奖励：接线验证与真实 doc_id 打通（T02 接线部分）

第 1 周 Tier2 已把 `evidence-wiring.md` 的三个丢失点改成代码并落地：`ray_trainer_ts` 把 `extra_info` 条件性带入 `gen_batch`、`run_llm_loop_tree_search` 逐叶写 `visible_doc_ids`、`RewardManager` 读 `evidence_alpha` 并调 `compute_evidence_reward` + `combine_score_with_evidence`。本周工作是：打通真实语料的稳定 `document.id`（让 `visible_doc_ids` 真正有值）、在真实 rollout 上验证两题不串题与覆盖率非零、完成 alpha 消融。

### 4.1 三个丢失点的落地状态与本周收尾

| # | 位置 | Tier2 已落地 | 本周收尾 |
|---|---|---|---|
| 1 | `verl/trainer/ppo/ray_trainer_ts.py`（`batch.pop`） | 把 `sample_id` / `supporting_doc_ids` / `evidence_status` 从 `extra_info` 提取后随 `gen_batch.repeat(k, interleave=True)` 传播 | 真实 rollout 确认证据字段到达树内 `reward_fn` |
| 2 | `search_r1/llm_agent/generation_ts.py`（`_passages2string` / `run_llm_loop_tree_search`） | `_batch_search` 用 `extract_doc_ids_from_retrieval` 提取 `document.id`，逐叶写 `visible_doc_ids` 到 `final_output`；按截断后 observation 判定「截断后可见」 | 真实语料返回稳定 `document.id` 后覆盖率才能非零（§3） |
| 3 | `verl/trainer/main_ppo_format_ts.py`（`RewardManager.__call__`） | 增加 `evidence_alpha`（默认 0，读 `reward_model.evidence_alpha`），逐条调 `compute_evidence_reward` 并 `combine_score_with_evidence` | 补两题不串题断言与 alpha 消融 |

### 4.2 关键不变量（已在第 1 周锁定）

- `final_node_list` 按 root 分组（每 root 的 `sample_leaf(k)` 叶子连续），与 `gen_batch.repeat(k, interleave=True)` 的 `np.repeat` 行序对齐；`final_output.batch['responses']` 由 `final_node_list` 的 `node.responses` 拼成。证据字段第 `i` 行对应 `final_node_list[i]`，`i // k` 定位根样本。
- 「截断后可见」语义：按 `_process_next_obs` 截断到 `max_obs_length` 后的 observation 判断哪些文档仍完整保留；不能用「后端返回了几个」替代。这是接线最细的部分。
- 证据奖励与 Recall@K 分别基于「模型可见集合」与「检索返回集合」留档，不混用（架构 §2.1）。

### 4.3 证据权重命名与选值流程

证据权重统一使用代码配置名 `evidence_alpha`（即架构 §2.1 的 `evidence_weight` / α）。选值流程：`ppo_trainer.yaml` 的 `reward_model.evidence_alpha`（默认 0.0）→ 训练入口 `main_ppo_format_ts.py`（约 209 行）`config.reward_model.get('evidence_alpha', 0.0)` → `RewardManager.__init__(evidence_alpha=...)`。默认 0.0 关闭证据项；`0.1` 仅作接线验证值；正式 E3 取值由 §7 消融确定。

### 4.4 验证与接线验收

- 用两个金标准证据不同、分支检索结果不同的混合样本验证「样本 → 树 → 叶子 → 奖励」逐项对应；交叉替换两题证据会使预期分数改变且不串题（架构 §2.1 / 第 1 周 Day 6 验收）。
- 无可靠 gold 时关闭证据项（`eligible=False`）并记录覆盖范围。

交付：真实 rollout 验证记录（两题不串题、覆盖率随 doc_id 打通而非零）+ 更新后的 `evidence-wiring.md`（蓝图 → 已落地状态）。

## 5. 实验配置：E1/E2/E3 公平对照

架构 §3 要求 E1/E2 公平对照不能直接用仓库默认脚本。本周正式训练前统一以下可统一项，并记录无法统一的算法相关参数：

| 参数 | 统一设置 |
|---|---|
| 模型起点 | 固定 Qwen2.5-3B 具体 revision（三实验相同） |
| 学习率 / 优化器 / warmup | 统一（承接第 1 周 smoke 起步 1e-6，正式值由小规模检查校准） |
| 有效 prompt batch / mini/micro batch | 统一，满足 GPU 与分组约束 |
| KL 设置 | 统一 |
| 生成长度 / 总上下文上限 | 统一 |
| 数据顺序与 epoch | 统一（同 seed 数据顺序） |
| 训练与生成预算停止规则 | 统一，预先固定 |

算法相关差异（记录，不强行抹平）：

- 普通 GRPO：`rollout.n_agent` 同题采样数（实际轨迹条数 = 同题采样数 × batch）。
- Tree-GRPO：`ts_m/ts_n/ts_l/ts_k`（M/N/L/K），其中 K 是每树最终抽样叶子数，不等于独立轨迹数或物理生成成本。

五组实验中本周只跑 Wiki 阶段的三个（E4 跳过 Wiki、E0 已在第 1 周完成）：

| 实验 | Wiki 阶段 | 说明 |
|---|---|---|
| E1 | GRPO，答案奖励 | 普通 GRPO 基线 |
| E2 | Tree-GRPO，答案奖励 | 验证树式方法 |
| E3 | Tree-GRPO，答案＋证据奖励 | 完整项目方案，`evidence_alpha` 由消融确定 |

成本口径（架构 §3）：分别报告物理新生成 token、逻辑搜索次数、实际外部请求（含重试）、GPU 时间；共享前缀不重复计入真实生成成本。轨迹条数相同不代表成本相同。

## 6. 训练、checkpoint 与评测口径

### 6.1 训练与 checkpoint

- 先小规模检查（128/64 或 1K 子集，10–20 步），确认无 NaN/Inf、证据奖励 alpha 生效、优势分组无串题、checkpoint 可导出后，再启动 12K 正式训练。
- 每个实验训练结束保存 checkpoint 并导出 HF（复用 `export_ckpt.sh` + 已修复的 `model_merger.py`）。三个 Wiki checkpoint（E1/E2/E3）是第三周 Web 迁移的输入，须在 manifest 中记录父 checkpoint、`global_step`、配置哈希与随机种子。
- 两阶段衔接口径沿用第 1 周：Stage 2 默认新建优化器，reference 初始化自 Stage-1 actor；只加载权重写「warm start」，不得称精确恢复。精确续训支持范围调查不在本周扩展。

### 6.2 评测指标

| 指标 | 口径 | 说明 |
|---|---|---|
| 多跳 EM | 每题题级 EM 平均，独立于训练奖励 | 阶段评测用 Wiki dev；正式 Multi-Hop Avg EM 用锁定测试集 |
| Recall@K | 按轨迹检索返回顺序累计、按 doc_id 去重、取前 K，算 gold 覆盖率 | 训练与证据奖励的「模型可见集合」分开留档；K=5、10 |
| 证据可映射比例 | `mapped` / 有效样本比例 | 报告有效样本数与 partial 单独统计 |
| 成本 | 物理生成 token、搜索次数、API 成本、GPU 小时 | 训练与推理分开 |
| 稳定性 | 无效调用率、超时率、重复搜索率、达预算上限比例 | 固定分母与错误分类 |

### 6.3 预算与成本统计（T10）

正式训练补齐第 1 周缺失的真实统计：逻辑搜索次数与外部请求次数（含重试）分开、缓存命中单独计、共享前缀不计入物理新生成 token。第 1 周发现的单卡 FSDP checkpoint ≈36G/step 磁盘压力在正式 8-GPU 分片下不复现，但仍须在日志记录 checkpoint 体积与磁盘占用。

## 7. 七天任务与逐日验收

### Day 1：语料、索引与映射管线就绪

任务：固定 Wikipedia 语料版本；建立 E5 + FAISS 索引；实现 `evidence_map_fn`（title→doc_id）；启动 `/retrieve` 并跑批量/空结果/非法输入契约测试。

交付：索引清单、`/retrieve` 契约测试结论、`evidence_map_fn` 代码。

验收：批量返回顺序稳定、doc_id 可回查正文、同一冻结语料重复查询可复核。

### Day 2：12K 数据生成与证据映射报告

任务：跑 `build.py` CLI 生成 12K train + dev parquet 与 manifest；产出可映射率 / partial / 歧义 / 语料缺失报告；至少 20 条人工核对。

交付：`train.parquet` / `dev.parquet` / `manifest.json`、证据映射小样本报告。

验收：`index` 全局唯一且 `index ↔ sample_id` 一一对应；train/dev 无泄漏；支持文档不进入 prompt。

### Day 3：证据接线验证与真实 doc_id 打通

任务：接线代码已在第 1 周落地；本周打通真实语料 `document.id`，在真实 rollout 验证两题不串题与覆盖率非零；本地用 `test_tree_evidence_wiring.py` 扩展断言。

交付：真实 doc_id 打通记录 + 真实 rollout 验证记录。

验收：树内回调收到每条叶子正确的金标准与实际可见文档 ID；交叉替换两题证据使预期分数改变且不串题；真实覆盖率非零。

### Day 4：小规模检查

任务：用子集跑 E1/E2/E3 各 10–20 步，检查稳定性；验证 `alpha` 生效、优势分组、checkpoint 导出；校准正式超参（LR、batch、生成长度）。

交付：小规模检查报告、正式实验配置冻结。

验收：三实验小规模均无 NaN/Inf、有非零更新、checkpoint 可导出重载。

### Day 5：正式 E1/E2/E3 Wiki 训练

任务：12K 数据，按冻结配置启动三个 Wiki 阶段训练（E4 跳过 Wiki）。

交付：三个训练的配置、日志、metrics。

验收：训练稳定运行，预算停止规则生效，成本统计齐全。

### Day 6：训练完成与 checkpoint 交付

任务：三实验训练收尾；保存并导出三个 Wiki checkpoint（HF）；记录父 checkpoint 与随机种子。

交付：三个 Wiki checkpoint 清单 + 导出模型。

验收：三个 checkpoint 均可导出、可重载，供第三周 Web 迁移使用。

### Day 7：评测、消融与第三周衔接

任务：用 Wiki dev 评测三个 checkpoint 的 EM 与 Recall@K；完成奖励消融（E2 vs E1 树收益、E3 vs E2 证据收益）；预注册 Multi-Hop Avg EM 测试集 manifest（HotpotQA/2Wiki/MuSiQue/Bamboogle 的题目 ID、数据版本、评分器）；写周报告与第三周资源估算。

交付：EM/Recall@K 表、消融结果、测试集 manifest、week2-report.md。

验收：消融结论可追溯到逐题结果；未提升的指标如实报告；第三周 Web 迁移输入（三个 Wiki checkpoint + 测试集 manifest）齐备。

## 8. 必要测试与门禁

### 8.1 第 1 周遗留门禁

第 1 周 T01–T10 中，本周须转 ✅ 的：T02（数据映射 + 真实 doc_id 打通）、T03（真实索引契约测试）、T05（接线验证 + alpha 消融）、T10（真实预算统计）。已 ✅ 的（T01/T04/T06/T07/T08/T09）沿用。

### 8.2 本周 Wiki 实验门禁

| 编号 | 验证对象 | 通过条件 |
|---|---|---|
| E01 | 12K 数据与证据映射 | index 全局唯一；可映射率已报告；gold 文档冻结语料一致性已核对 |
| E02 | 证据接线验证与 doc_id 打通 | 样本→树→叶子→奖励一一对应；两题不串题；截断后可见性与 retrieval_hit 判定正确；真实覆盖率非零 |
| E03 | E1/E2/E3 配置统一 | 可统一超参一致；算法相关差异已记录；预算停止规则固定 |
| E04 | 小规模稳定性 | 三实验小规模无 NaN/Inf、有非零更新 |
| E05 | 三 Wiki checkpoint | E1/E2/E3 训练完成、可导出、可重载 |
| E06 | 评测口径 | EM 独立于训练奖励；Recall@K 与证据奖励集合分开留档 |
| E07 | 消融可复现 | E2 vs E1、E3 vs E2 结论可追溯到逐题结果 |

只测影响训练正确性与实验可信度的边界，不为简单目录或文档改动增加测试。

## 9. 阻塞与降级策略

| 阻塞 | 处理 | 对验收的影响 |
|---|---|---|
| FlashRAG 语料/索引未到位 | 接线代码已落地，先做本地断言与 alpha 消融准备；数据与正式训练顺延 | T02 数据部分、E05 保持未完成，不伪报 |
| 证据映射率过低 | 主训练先 `evidence_alpha=0`，排查 title 定位与切块；E3 证据项延后 | 不宣称证据奖励有效 |
| 正式训练 OOM / 不稳定 | 减 micro batch / 生成长度，调整 offload；保留实际配置 | 与 smoke 结果分界，不混用 |
| 评测检索预算不足 10 文档 | 增大轮数或 top-k，去重后如实统计真实候选数 | 不补齐、不选择性剔除 |
| 时间或算力不足 | 先缩小数据跑通 E1/E2/E3 全对照，再扩大 | 不删除 GRPO 对照却保留「优于 GRPO」结论 |

发生延期时，优先保证「E1/E2/E3 三个 Wiki checkpoint + 证据接线验证 + 消融可复现」，缩减样本量与步数，不删正确性检查。

## 10. 第二周完成清单

- [ ] 12K Wiki 数据（NQ/HotpotQA/2Wiki/MuSiQue）与独立 dev 生成，manifest 含抽样种子。
- [ ] 证据映射完成，可映射率 / partial / 歧义 / 语料缺失已报告。
- [ ] 树内证据奖励接线验证通过（两题不串题、真实覆盖率非零，接线代码已在第 1 周落地）。
- [ ] E1/E2/E3 配置统一，算法相关差异与预算停止规则已记录。
- [ ] 小规模检查通过，正式超参冻结。
- [ ] E1、E2、E3 三个 Wiki checkpoint 训练完成、可导出、可重载。
- [ ] EM / Recall@K / 证据可映射比例评测表完成，口径正确。
- [ ] 奖励消融（E2 vs E1、E3 vs E2）完成并可追溯到逐题结果。
- [ ] Multi-Hop Avg EM 测试集 manifest 预注册（题目 ID、数据版本、评分器）。
- [ ] 成本统计齐全（物理生成 token、逻辑搜索、外部请求、GPU 时间）。
- [ ] week2-report.md 完成，第三周 Web 迁移输入齐备。

全部门禁通过后进入第三周 Web 迁移。本周成果表述为「完成 Wiki 阶段 E1/E2/E3 三个实验与证据奖励，得到三个 Wiki checkpoint 及 EM/Recall@K 与奖励消融结果」；任何性能提升、证据召回改善或成本优势，须待正式对照结果产出后再写入简历。

## 11. 工单拆分与依赖顺序

| 工单 | 范围 | 依赖 | 完成证据 |
|---|---|---|---|
| W2-01 | 冻结语料、FAISS 索引、`/retrieve` 契约测试 | FlashRAG 语料/索引 | 索引清单、契约测试结论 |
| W2-02 | `evidence_map_fn` + 12K 数据 + manifest | W2-01 | parquet、manifest、映射报告 |
| W2-03 | 证据接线验证 + 真实 doc_id 打通 + alpha 消融 | W2-02（数据字段） | 两题不串题断言、覆盖率非零记录 |
| W2-04 | E1/E2/E3 配置统一与小规模检查 | W2-02、W2-03 | 冻结配置、小规模检查报告 |
| W2-05 | 正式 E1/E2/E3 训练与 checkpoint | W2-04 | 三个 Wiki checkpoint 清单 |
| W2-06 | 评测与消融 | W2-05 | EM/Recall@K 表、消融结果 |
| W2-07 | 测试集 manifest 预注册与周报告 | W2-05、W2-06 | manifest、week2-report.md |

## 12. 第三周衔接与资源估算

第三周（Web 迁移）输入：三个 Wiki checkpoint（E1/E2/E3 的 HF 导出）+ 预注册的 Multi-Hop Avg EM 测试集 manifest。本周须为第三周留下：父 checkpoint 记录、配置哈希、随机种子、评测评分器代码，使 Web 迁移与最终评测能直接复用同一评分器与题目 ID。

资源估算沿用第 1 周 §11 方法，用本周正式训练实测的稳态秒/step 外推：

`GPU 小时 ≈ 稳态秒/step × 目标 step 数 × GPU 数 / 3600`

训练规模由样本数、epoch、有效 prompt batch 与 sampler 共同决定，不能仅凭「12K 数据」推算；检索延迟与长轨迹使小样本线性外推不可靠。报告乐观/基准/保守三档，正式排期采用测量结果，并把检索服务占用、模型加载、checkpoint 写入、评测和重试单独列出。

## 13. 参考底座

同第 1 周方案 §12。本周主要复用第 1 周已核对的代码结构：`researchsearch/data/build.py`、`verl/utils/reward_score/evidence.py`、`search_r1/llm_agent/generation_ts.py`、`verl/trainer/ppo/ray_trainer_ts.py`、`verl/trainer/main_ppo_format_ts.py`、`scripts/merge_ckpt/model_merger.py`。源码行号以第 1 周快照为准，后续修改可能移动；定位时同时使用文件名和函数名。
