# ResearchSearch-RL：第 1 周工程闭环开发方案

版本：v1.5 · 2026-10-19（Web 工具范围固定为搜索摘要）  
项目根目录：`/Users/ct/code/算法/researchsearch`  
代码目录：`/Users/ct/code/算法/researchsearch/Tree-GRPO-main`  
文档位置：项目根目录 `ResearchSearch-RL-第1周工程闭环开发方案.md`。下文源码路径均相对于代码目录。  
核查范围：启动脚本、数据预处理、两类 rollout、奖励、优势计算、检索服务、验证及 checkpoint 保存/加载。已经确认代码存在；尚未安装依赖或执行 GPU/API 训练验证。遵循项目根目录 AGENTS.md。

项目总方案见[项目架构与实验设计](./ResearchSearch-RL-项目架构与实验设计.md)。本周 smoke 使用其中规定的现有分段奖励和独立证据项，正式算法对照按总方案统一超参、预算与评测流程。

## 1. 本周目标与边界

在已有 Tree-GRPO 源码上，完成可记录、可评测、可加载 checkpoint 继续训练的最小链路：

```text
固定模型与数据 → Wiki 检索 → 多轮 rollout → 分项奖励
→ GRPO / Tree-GRPO 参数更新 → checkpoint 保存
→ 模型导出与重新加载 → Web 检索 → 少量 RL 更新
→ 独立开发集评测 → 结果与成本归档
```

本周的成功标准是工程正确性，不要求准确率提升。正式 12K Wiki、2.2K Web 训练、四项 Web benchmark 全量评测与多随机种子实验安排在后续阶段。本周不默认做 SFT、不训练检索器、不增加 BGE 对照、不开发异步训练、不合成大规模 Web 难题。

默认保留现有 `<think>`、`<search>`、`<information>`、`<answer>` 协议。Web 使用已有搜索工具返回摘要，Agent 可以多轮搜索并改写查询。本周 smoke 与后续 E0–E4 正式对照使用相同的摘要工具协议、提示和预算。WebWalkerQA、GAIA-text、SimpleQA、BrowseComp 的结果标明“仅使用搜索摘要”，不称为完整网页浏览或网页正文级证据搜集。

计划按一名开发者约 5–7 小时/日组织；GPU 可用时长另计。GPU 型号、数量、显存、主机内存、磁盘和 API 配额尚未确认，不假设已有 4×A800。CUDA 训练需在兼容的 GPU 主机完成，本地目录仅用于代码和文档工作。

## 2. 已实现能力与实际开发缺口

第一周应在现有链路上完成配置、接线与验证。以下“已有”表示源码已实现，不代表已在目标服务器运行成功。

| 能力 | 已核对的源码 | 当前实现与本周动作 |
|---|---|---|
| 普通 GRPO | `train_multihopqa_grpo.sh`、`verl/trainer/main_ppo_format.py`、`verl/trainer/ppo/ray_trainer.py` | 已有完整训练循环；`n_agent=4` 经 batch.repeat 实现同题多轨迹。复用并缩小配置 |
| Tree-GRPO | `generation_ts.py:554`、`tree_node.py`、`ray_trainer_ts.py:127` | 已有树展开、前缀状态复用、叶子抽样；`adv_estimator=tree` 分别按问题 uid、tree_uid 求优势后相加。无需重写 |
| 树采样预算 | `generation_ts.py:625` 与启动脚本 | `ts_k` 是每棵树最终抽取的训练叶子数；启动脚本 M=2/N=2/L=1/K=3，最终叶子槽位为每题 M×K=6。不能把它等同于普通 GRPO 的 4 条独立生成成本 |
| observation mask | `generation.py`、`generation_ts.py`、两类 trainer 的 `_create_loss_mask` | 已有 info_mask 与 loss_mask；补正确性检查和日志，不从头实现 |
| 本地检索 | `search_r1/search/retrieval_server.py:327` | 已有批量 /retrieve、稠密/稀疏检索以及 document/score 返回；补稳定 ID 旁路记录 |
| Web 检索 | `search_r1/search/bing_search_server_redis.py` | 已有 Brightdata 请求、批量并发、超时重试、Redis 缓存及 JSON 持久化；核验凭据与服务可用性，补结构化元数据 |
| 奖励 | `qa_em_format.py:231`、`qa_f1_format.py:275`、两个 RewardManager | 已有 EM/F1、格式、答案字符串检索命中项。没有支持文档覆盖奖励；新增独立 evidence 项 |
| 数据加载 | `scripts/data_process/qa_search_train_merge.py:56`、`rl_dataset.py:120` | 预处理路径硬编码，且每个数据集的 index 从 0 重置；拼接后会出现同题分组 ID 冲突。Day 2 生成全局唯一整数 index，扩展来源/证据，并核验 repeat 和树分支对应关系 |
| 基础验证 | `ray_trainer_ts.py:564`、`ray_trainer.py` | 已有按 data_source 汇总奖励分数的验证；Tree trainer 验证使用普通链式 generation。补逐题纯答案指标与成本导出 |
| checkpoint | `ray_trainer_ts.py:750`、`fsdp_workers.py:645`、`fsdp_checkpoint_manager.py` | 已有模型/优化器/调度器等状态保存加载基础能力。trainer 的 fit 从 global_steps=0 开始，未看到完整恢复调度接线；本周优先验证权重迁移 |
| 模型导出 | `scripts/merge_ckpt/model_merger.py`、同目录转换脚本 | 已有合并器；包装脚本路径硬编码且依赖工作目录，需参数化 |

### 2.1 已确认的启动阻塞

1. 多跳启动脚本 DATA_DIR 为空，默认八卡；目标服务器配置尚未确认。
2. 多跳与 Web Tree 启动脚本使用 `$lr`，脚本内没有赋值；必须显式设置和预检。
3. Web Tree 默认模型为 Qwen2.5-14B，与项目 3B 目标不一致。
4. 数据处理器依赖 `/mnt/workspace/common/datas/FlashRAG_Dataset`；模型转换脚本也依赖作者机器路径。
5. 普通 GRPO 与 Tree 脚本 warmup 比例不同。正式算法对照应统一或说明差异，不能直接把两个默认脚本当作严格控制变量实验。
6. 两类 rollout 的 `_batch_search` 直接 requests.post(...).json()，未设置客户端超时与 HTTP 状态检查；服务端 Web 超时不能替代客户端保护。
7. `qa_search_train_merge.py` 对每个数据集重新生成 `index=0,1,...` 后拼接；普通和 Tree trainer 都将 `index` 赋给优势分组 `uid`。混合 batch 中重复 index 可串组，新增 `sample_id` 字段不会自动修复。
8. Tree trainer 传入树采样的生成 batch 只有 `uid`、`data_source`、`reward_model` 等字段；树采样在与原 batch 合并前调用奖励函数。仅在 parquet `extra_info` 新增支持文档 ID，树内奖励仍读不到它。

### 2.2 本周优先级

- P0：参数化启动、两算法更新、保存/导出/加载、真实 Web 单步迁移。
- P1：数据证据元信息、逐题日志、纯答案评测、工具错误分类、证据奖励接线。
- P2：精确断点恢复、多 provider 抽象、正式规模与多种子；本周记录摘要搜索接口与日志字段，正式 Web 训练沿用该工具范围。

不直接升级 veRL。先按本地 README 建立训练与检索环境，再记录成功组合；版本说明不等于已通过本机兼容验证。

## 3. 开发架构与文件职责

以下为拟新增目录，不表示它们已经存在。尽量保留上游代码结构，只在明确接口处改动。下表列出已核对的现有入口，实施时优先在这些位置扩展。

```text
Tree-GRPO-main/
  configs/researchsearch/       # 环境、算法、奖励、规模配置
  scripts/researchsearch/       # 预检、数据准备、启动、评测、导出
  researchsearch/
    data/                      # 统一格式、去重、证据映射
    tools/                     # Web 适配、工具日志、缓存
    rewards/                   # 证据覆盖与分项日志
    evaluation/                # 单轨迹评测与统计
  tests/researchsearch/         # 数据、奖励、mask、工具与导出测试
  docs/                        # 环境、运行说明、数据与指标定义
  artifacts/<run_id>/           # 配置、日志、轨迹、评测结果
```

大模型、Wikipedia、索引和 checkpoint 放在可配置的数据盘目录，避免提交到版本库。API 密钥通过环境变量注入，不进入配置快照和日志。

| 原有模块 | 本周操作 |
|---|---|
| `verl/trainer/main_ppo_format.py` | 核验作为普通 GRPO 启动入口的配置与奖励路由 |
| `verl/trainer/main_ppo_format_ts.py` | Tree-GRPO 入口，接入统一实验配置与日志 |
| `search_r1/llm_agent/generation*.py` | 工具超时、轨迹记录；保留文档 ID 与 agent-step 边界 |
| `search_r1/llm_agent/tree_node.py` | 检查 tree/node/parent 标识与前缀复用关系 |
| `verl/trainer/ppo/ray_trainer*.py` | 观察 reward、advantage、mask、保存与评测链路 |
| `verl/trainer/ppo/core_algos.py` | 阅读并测试已有优势计算；本周不重写算法 |
| `verl/utils/reward_score/qa_em_format.py`、`qa_f1_format.py` | 复核原有评分语义，避免格式惩罚重复叠加 |

## 4. 数据与接口约定

### 4.1 第一周数据规模

| 数据 | 建议规模 | 使用边界 |
|---|---:|---|
| 手工可验证 fixture | 10–20 条 | 解析、奖励、异常测试；不报告为模型能力 |
| Wiki smoke train | 128 条，来自 train split | 先以 HotpotQA 为主，跑通后纳入其余三类数据的小样本 |
| Wiki dev | 64 条，独立划分 | E0 和训练后开发评测，不能来自 smoke train |
| Web smoke train | 32 条，来源明确且有答案 | 工具与迁移训练；优先公开训练数据 |
| Web dev | 20–32 条，独立问题 | 初步质量与故障率观察，不推断正式 benchmark 得分 |

全量计划另生成 manifest：NQ 2K、HotpotQA 4K、2Wiki 3K、MuSiQue 3K；ASearcher 2K、WebDancer 200。第一周只确认来源、字段、可获得性，不要求下载整理全部数据。

ASearcher/WebDancer 核查须落实到实际可获取文件、版本、可用样本数、筛选规则和抽样种子，并记录与开发集及正式评测集的去重结果。2.2K 是目标规模，不能在来源核查前标记为数据已就绪。manifest 保留样本清单，逐条可追溯数据抽样种子；训练随机种子另行记录。

对问题做规范化去重，并保存原始样本 ID。源数据的 train/dev/test 不能随意混用。正式测试集锁定，本周只用开发集调试。合成 fixture 不得混入真实评测。

### 4.2 统一样本结构

沿用上游 parquet 主字段，拟扩展 `extra_info`：

```json
{
  "data_source": "hotpotqa",
  "prompt": [{"role": "user", "content": "模板与问题"}],
  "ability": "fact-reasoning",
  "reward_model": {
    "style": "rule",
    "ground_truth": {"target": ["标准答案", "合法别名"]}
  },
  "extra_info": {
    "sample_id": "hotpotqa:原始ID",
    "index": 12345,
    "split": "train",
    "source_revision": "固定版本",
    "environment": "wiki",
    "supporting_doc_ids": ["wiki:doc1", "wiki:doc2"],
    "evidence_status": "mapped",
    "corpus_version": "固定语料标识"
  }
}
```

`index` 是拼接后的全局唯一整数训练分组 ID，不是各数据集内部行号；原始行号可另记为 `source_index`。按固定 manifest 顺序给每条不同题目分配，检查 `index ↔ sample_id` 一一对应。相同题目的 GRPO 重复采样、Tree 多棵树和叶子应保留同一个 `uid=index`，不同题目不能共用 uid，即使跨数据集原始行号相同。

必须验证 dataset loader、batch repeat、tree expansion 到 reward callback 的整条路径。当前 Tree trainer 在生成前 `pop` 的非张量字段中没有 `extra_info`，而树采样会先调用奖励函数；因此需要把 `sample_id`、`supporting_doc_ids`、`evidence_status` 及必要的语料版本从根样本带入生成 batch，按采样叶子的实际顺序写入奖励输入。每个叶子的整条轨迹须携带实际进入观察的文档 ID，包含共享前缀和分支后的检索结果；奖励回调逐叶核对 `uid/sample_id/gold/visible_doc_ids`。只在 parquet 中加字段不能视为奖励接通。答案、支持文档 ID 不能进入模型 prompt 或工具观察。

证据状态至少区分 mapped、partial、unmapped、not_available。主证据召回只在全部金标准支持文档可映射的样本上统计，同时报告有效样本数和比例。partial 样本单独报告，不通过缩小分母伪造高召回。

### 4.3 检索协议

第一周统一 Agent 搜索调用协议与日志字段，兼容下述两种后端返回格式；Wiki 文档 ID、Web 结果原始排名、错误状态及实际请求计数必须保留。Web 模型观察继续使用现有格式化搜索摘要；如果服务提供 URL，则仅作为旁路日志元数据保存。所有 E0–E4 Web 对照使用同一搜索工具、摘要截断规则与调用预算。

现有 rollout 发出的请求形状：

```json
{"queries":["查询内容"],"topk":3,"return_scores":true}
```

Wiki 返回形状如下（稳定 id 需根据实际语料核验/补齐）：

```json
{"result":[[{"document":{"id":"wiki:123","contents":"标题\n正文"},"score":0.8}]]}
```

批量请求与返回外层顺序一一对应。模型看到的是标题与内容；证据评分使用旁路保留的 ID，不能依赖字符串中的 Doc 1、Doc 2，因为这些编号会在不同搜索中重复。

当前 Web 服务返回 `{"result":["Page 1: 摘要…"]}`，为字符串列表；现有 `_passages2string` 能处理该形状。Web 服务仅从 organic 条目提取 description，未保留 URL/标题；max_results 是服务启动参数，请求中的 topk 不参与现有 endpoint 处理。第一周保留兼容返回，同时新增结构化旁路记录和实际结果数统计，不能只修改客户端 topk 就宣称 Web 预算已改变。

Web 无原生相似度时不得伪称为向量分数；若兼容层需要数值字段，使用占位并在元数据标记其含义，保留原始 rank。查询、抓取时间、provider、缓存键写入日志；若服务提供 URL，可作为旁路元数据保存。

当前 Web 服务已有服务端超时和重试，但最终失败以普通文本返回，缓存键仅使用 query。补充 status/error_code 和 provider、topk、版本组成的缓存键；沿用 Redis/JSON 实现。客户端与服务端共同约束总重试预算，避免嵌套重试放大请求数。

超时与重试初始建议：单次请求 30 秒，最多重试 2 次，指数退避；对不可重试的鉴权错误立即失败。区分真实零结果、限流、网络失败和非法返回。健康预检失败时禁止启动正式训练；运行中系统错误导致的轨迹标记无效，不静默当作答案错误计入训练，补样或终止该批次并记录原因。

### 4.4 奖励与证据口径

本周先保持原有答案与格式奖励语义，只记录每个分项。证据项通过开关启用：

`R_total = R_existing + alpha * R_evidence`

`R_existing` 保留上游分段评分语义：正确且格式合法得 1，正确但格式不合法扣结构分；错误但格式合法可能获得结构分。它不等价于简单的 EM 减格式惩罚。本周不同时重写这套规则，以免奖励实验混入额外变量。`retrieval_score` 仅判断 information 中是否包含答案字符串，继续固定为 0；新增独立 evidence_weight。

`R_evidence = |实际观察到的去重支持文档 ∩ gold| / |gold|`。

这里“观察到”要求文档在截断后仍进入 Agent 观察；后端检索到了、但内容被截掉的文档另记 retrieval hit，不能算 Agent 已读证据。第一周采用文档级覆盖，不声称句子级证据充分性或引用正确性。

Tree 奖励输入以最终 `final_node_list` 的叶子顺序为准；该顺序必须和 `final_output` 每一行及 `original_scores[i]` 一一对应。根样本的金标准证据沿每条叶子的父链传递，实际可见文档 ID 从该叶子的完整轨迹提取；不能在树内奖励执行后再从重复的原 batch 追加证据字段。

alpha=0 为默认；alpha=0.1 可作为接线验证值，不视为最终超参。无可靠 gold 时关闭该项并记录 eligible=false。重复搜索同一证据不重复给分。Web 阶段保持 alpha=0。

Recall@K：按搜索轮次与每轮原始排名依次合并，按稳定 doc_id 去重，取前 K 个，与 gold 比较。若实际返回不足 K，不补齐；记录真实候选数量。三轮 top-3 最多九个不同候选，因此该 smoke 配置不能支持有意义的 Recall@10 上限比较；本周先验证计算，正式评测另固定足够的检索预算。

正式 Recall@10 的预算须允许至少 10 个不同文档；预算容量达标后仍须记录去重后的实际数量，不能因重复、空结果或提前结束而选择性剔除样本。该指标基于检索返回集合，证据奖励基于截断后模型可见集合，两者分别保存以供复核。

## 5. 配置、checkpoint 与运行记录

### 5.1 初始 smoke 配置

| 参数 | 起步设置或规则 |
|---|---|
| 模型 | 固定 Qwen2.5-3B；base/instruct 与 revision 明确，不能混用 |
| 优化器学习率 | 显式设置 1e-6，作为初始值，不使用未定义 shell 变量 |
| Tree 参数 | M=2、N=2、L=1；对应 ts_m/ts_n/ts_l；ts_k=3 是每树最终抽样数，另行记录 |
| Wiki 工具预算 | 最大 3 次搜索、topk=3 |
| 更新步数 | 先 2 个 step 验证，再 10–20 个 step 检查稳定性 |
| prompt batch | 尝试 8–16；须满足实际 GPU、mini/micro batch 与分组约束 |
| 生成长度 | 先约 256–512 token/轮，总上下文上限独立控制 |
| checkpoint | 在 smoke 结束时必存，不依赖可能高于 smoke 步数的默认保存间隔 |
| 日志 | console + 本地文件；在线追踪可选 |

不能把 `data.max_response_length` 直接当作整条多轮轨迹预算。开发时追踪实际使用位置，分别记录每轮生成上限、总上下文、最大轮数和最大总生成 token。普通 GRPO 当前通过 rollout.n_agent=4 对 batch 重复生成，确实影响同题组大小；Tree 路径使用 ts_m 与 ts_k。记录两者实际轨迹和新生成成本，不能按配置名称推断公平性。

以上均为待验证起步配置，不是已证明能在某型号 GPU 上运行的参数。OOM 时先减 micro batch、生成长度和 batch，再调整 offload；记录变更，不悄悄换模型或算法。

正式 E1/E2 不能直接沿用默认脚本宣称公平：须统一可统一的学习率、优化器、warmup、有效 prompt batch、KL、生成长度及数据顺序，完整记录 GRPO 同题采样数和 Tree 的 M/N/L/K。固定生成与搜索预算停止规则，报告真实生成 token、搜索次数、GPU 时间及预算偏差；所有组沿用同一链式推理评测流程。

### 5.2 保存、加载、恢复是三件事

1. 保存：确认所有 rank 分片、模型配置和 tokenizer 可定位，记录文件完整性。
2. 阶段迁移：把 Wiki actor 导出为可加载模型；新进程加载该模型，在 Web 环境进行至少一次更新。Stage 2 默认新建优化器，reference 初始化自 Stage-1 actor，配置中明确记录，后续所有两阶段对照一致。
3. 断点恢复：恢复优化器、调度器、global step、随机状态及数据进度。底层 FSDP manager 已有加载接口，但已核查的两个 trainer 没有完整的恢复调度，fit 会重置 global_steps；需另外接入训练进度及数据状态。只有逐项验证才标记“支持精确续训”。只加载权重应写“warm start”，不得称精确恢复。

第一周必须完成第 1、2 项；第 3 项完成支持范围调查和结果记录，不能伪报成功。

### 5.3 单次运行产物

每个 run_id 保存：解析后的完整配置、源码版本或源码快照哈希、依赖清单、数据 manifest 哈希、模型 revision、父 checkpoint、随机种子、硬件信息、日志、metrics.jsonl、trajectories.jsonl、evaluation.jsonl、summary.json、checkpoint 清单。

轨迹至少包含 sample_id、run_id、tree_id、node_id、parent_id、step_id、query、文档 ID、工具状态、缓存命中、终止原因、最终答案、奖励分项及计数。Agent 逻辑搜索次数、外部 API 请求次数（含重试）、缓存命中次数分开统计。

Web 轨迹记录每轮查询、搜索摘要及原始排名、进入模型观察的截断范围、后续查询、工具状态与最终答案；若服务提供 URL，可记录在旁路日志中。正式 E2/E4 迁移比较在相同题目上同时展示最终分数和摘要搜索行为，并归档成功及失败案例。

树分支复用的前缀不能重复计入实际生成成本。分别记录物理新生成 token 与展开后的训练 token；API 费用按实际外部调用计。GPU 小时按占用 GPU 数 × 时间统计，训练与评测分开。

## 6. 七天任务与逐日验收

### Day 1：确认环境、消除启动阻塞

任务：
1. 记录 GPU 主机硬件、驱动、CUDA、磁盘与网络条件；区分训练和检索资源。
2. 本地源码目录没有 .git；保存源码快照哈希与来源，后续建立版本管理时保留该基线，不杜撰上游 commit。
3. 按源码推荐建立独立训练/检索环境，记录实际成功安装版本。
4. 参数化数据、模型、学习率、GPU 数、日志目录和 Ray 地址；提前创建目录，启用参数缺失检查与 pipeline 失败传播。
5. 检查 Ray runtime_env 是否正确分发本地代码和环境变量，worker 能否访问模型、数据和检索服务。
6. 跑 tokenizer、短文本生成及一个 optimizer step 的基础检查；提前发起一个真实 Web API 请求验证鉴权与配额。

交付：environment.md、依赖锁定记录、preflight 脚本、参数化 smoke launcher。

验收：缺模型/缺数据/缺学习率时能在提交 GPU 任务前失败；解析配置中不存在空路径；模型能生成。基础 optimizer 检查不等同于已跑通 RL。

### Day 2：Wiki 服务与数据打通

任务：
1. 准备 E5、语料、FAISS 索引，核验向量维度、语料条数与索引条目映射。
2. 若全量索引下载未完成，先用固定小语料建索引，并明确标记 tiny-corpus，不与正式成绩比较。
3. 启动 /retrieve，测试单条、批量、中文/英文字符、空结果、非法输入与服务不可用。
4. 整理 Wiki 128/64 样本；补 sample_id、source_revision 与 evidence_status。修改混合数据预处理：按固定 manifest 顺序在所有生成的 Wiki 划分中为每条不同题目分配不重复的整数 `extra_info.index`，保留数据集内原始行号，并记录训练/开发划分的 ID 范围与映射。
5. 对至少 20 条带支持文档的样本做人工映射检查，报告可映射率和歧义案例。

交付：train/dev parquet、manifest、retrieval-contract.md、证据映射小样本报告。

验收：批量结果顺序稳定、doc_id 可回查正文、同一冻结环境重复查询可复核；训练/开发 ID 无交集；混合数据 `index` 无碰撞且每个 index 只映射一条 `sample_id`；支持文档元数据不会进入 prompt。

### Day 3：Agent rollout、E0 与奖励测试

任务：
1. 复用已有验证路径运行带工具 E0，同时设置 val_before_train=true 与 val_only=true；后者只在前者分支内生效。补逐题导出，并确认未进入优化器更新。
2. 记录多轮 query、观察、最终答案、结束原因和工具成本。
3. 对正确答案、错误答案、别名、格式错误、重复证据、无 gold、截断证据实现奖励测试。
4. 在生成与训练数据之间检查 assistant token 和 observation token 的 mask。
5. 对超时、非法 JSON、限流和空结果执行故障注入，检查不会无限等待或无限重试。

交付：E0 结果、奖励/工具测试、完整轨迹样例。

验收：至少一条真实轨迹完成两轮搜索并正常结束；未训练模型若做不到，用 fixture 验证协议但如实记录模型失败。不得用固定成功轨迹冒充真实 rollout。

### Day 4：普通 GRPO 更新闭环

任务：
1. 运行 2 步 ordinary GRPO，再视稳定性扩展至 10–20 步。
2. 用至少两个来源且原始行号相同的样本组成一个 batch，检查 `uid` 由全局 index 派生；每个优势组只含同一道题，再检查 reward、advantage、log probability、KL 与损失是否有限。
3. 确认存在有奖励差异的组；全同奖励导致零优势时记录原因，不伪造 reward。
4. 验证 optimizer step 完成且抽样参数发生有限变化；确认生成侧在下一轮使用更新后的策略。
5. 保存 checkpoint 并检查分片。

交付：w1-wiki-grpo-smoke 的配置、轨迹、指标与 checkpoint 清单。

验收：无 NaN/Inf、非 padding 的模型输出参与 loss、observation 不参与策略梯度；至少一次有效非零更新。reward 上升不是硬验收条件。

### Day 5：Tree-GRPO 更新与结构校验

任务：
1. 从与 Day 4 相同的起点运行 M=2/N=2/L=1 的 smoke。
2. 导出树结构，核对分支从完整 agent-step 边界展开，前缀与 observation 状态一致。
3. 检查 adv_estimator=tree 的 uid/tree_uid 两组标准化后相加；不要误把 core_algos.compute_tree_grpo_outcome_advantage（tree_2norm 路由）当默认算法。用固定数值验证零方差、单元素及混合奖励组。
4. 检查共享前缀的 token、mask、log probability 对齐，不把逻辑复用误当实际重生成。
5. 比较两种算法的工具/生成成本记录，不以十几步小样本比较宣称算法优越。
6. 验证根样本字段随生成 batch 进入树采样；检查最终叶子顺序与样本 ID 的对应关系，给 Day 6 树内证据奖励接线建立固定测试样例。

交付：w1-wiki-tree-smoke、树结构样例、算法与 mask 检查报告。

验收：发生真实分支采样、成功参数更新、两种算法都产生可读取的统一日志。

### Day 6：证据接线与 Web 迁移闭环

任务：
1. 先在已有轨迹离线计算证据奖励；把根样本证据字段和每条叶子实际可见文档 ID 按最终叶子顺序送到树内 `reward_fn(final_output)`。用两个金标准证据不同、分支检索结果不同的混合样本验证“样本 → 树 → 叶子 → 奖励”逐项对应，确认去重、共享前缀及截断处理后，再启用 alpha>0 做 1–2 次 Wiki 更新。
2. 复用 scripts/merge_ckpt/model_merger.py；包装脚本通过自身目录定位合并器，并参数化输入输出。加载导出模型做确定性推理一致性检查；同精度允许约定数值误差，不要求跨后端逐 bit 一致。
3. 优先运行已有 Redis Web 服务并接入 /retrieve，准备独立 train/dev 小样本；补缓存命名空间、真实结果数量、URL/rank 旁路元数据与结构化失败状态。
4. 从 Wiki actor 启动新的 Web 训练进程，alpha=0，完成至少一次有效更新并保存。
5. 记录 parent checkpoint、优化器重置策略和 reference 来源。

交付：证据奖励接线结果、导出模型清单、w1-web-transfer-smoke、Web 工具轨迹。

验收：树内回调收到每条叶子正确的金标准与实际可见文档 ID，交叉替换两题证据会使预期分数改变且不会串题；不是从基础模型误启动；真实 Web 调用进入 rollout；Wiki checkpoint → Web 更新 → 保存路径贯通。Mock Web 只能证明离线接口，不满足本项完整验收。

### Day 7：统一评测、重新加载与交付

任务：
1. 使用相同开发集、提示、工具预算与推理参数评测 E0、Wiki GRPO、Wiki Tree 和 Web 迁移模型。沿用链式推理作为共同评测策略，不能让 Tree 模型额外享受树搜索推理预算。纯 EM/F1 与训练奖励分别导出，不直接把 val/test_score 当准确率。
2. 测试 checkpoint 加载后的新进程运行；调查并记录精确续训支持范围。
3. 汇总答案分数、格式有效率、检索/系统错误、成本及证据指标有效样本数。
4. 抽查至少 10 条失败轨迹，按查询失败、证据缺失、推理错误、格式错误、工具故障分类。
5. 用实际 step 时间、生成 token、工具延迟估算后续运行成本，并列明小样本外推限制。
6. 写出从空环境到 smoke 结果的运行说明，完成交付检查表。明确第 3 周正式 Web RL 继续使用搜索摘要工具，以及四项等权 Multi-Hop Avg EM 的 HotpotQA、2Wiki、MuSiQue、Bamboogle 测试 manifest（数据版本、split、题目 ID、评分器）准备任务。核对实际 optimizer 更新次数：当前循环在步数递增后判断结束，checkpoint 名称中的 global_step 不应直接视为已完成更新数。

交付：week1-report.md、结果表、失败案例、运行说明、第二周任务与资源估算。

验收：另一名开发者可以根据文档找到输入、配置、启动方式与输出；结果均能定位到逐题轨迹。本周报告明确写“工程 smoke”，不填截图中的提升数值。

## 7. 必要测试与门禁

| 编号 | 验证对象 | 通过条件 |
|---|---|---|
| T01 | 数据隔离与分组 ID | train/dev ID 和规范化问题无意外重叠；混合数据每题具有唯一整数 index，跨数据集同原始行号不串 uid 组 |
| T02 | 证据元数据 | 根样本经 loader、repeat、tree expansion、叶子采样进入树内奖励回调后仍对应正确金标准；实际可见文档 ID 与叶子轨迹一一对应 |
| T03 | 检索契约 | 批量顺序、文档 ID、空结果与错误状态正确 |
| T04 | 格式与答案奖励 | 手工样例结果符合明确规则，惩罚不重复 |
| T05 | 证据奖励 | 重复文档不加分，缺 gold 不除零，截断结果不冒充已读 |
| T06 | loss mask | 工具观察和 padding 不计策略损失；模型生成正确参与 |
| T07 | 分组优势 | 同题分组正确，树内/树间不串组，零方差有限 |
| T08 | 权重更新 | 有非零有效更新，下一轮 rollout 加载新权重 |
| T09 | checkpoint | 文件完整，可导出、可加载、可用于下一阶段更新 |
| T10 | 预算 | 达到上限终止；缓存/重试与逻辑调用分别统计 |

只测试会影响训练正确性与实验可信度的边界，不为简单目录创建或文档改动增加测试。

## 8. 阻塞与降级策略

| 阻塞 | 处理 | 对验收的影响 |
|---|---|---|
| GPU 尚未提供 | 先做数据、工具、奖励、配置检查 | 不能宣称训练闭环完成 |
| 全量 Wiki 下载慢 | 固定 tiny-corpus 跑 smoke | 可以完成接口验证，正式检索仍待完成 |
| 依赖冲突 | 固定成功组合并记录；训练和检索隔离 | 不盲目升级整套框架 |
| OOM | 减 micro batch/长度，检查总上下文和 offload | 保留实际配置；不得与大预算结果混用 |
| 证据映射差 | 主训练先 alpha=0，排查标题与切块 | 不宣称证据奖励有效 |
| Web API 不可用 | 先做缓存回放和 mock，继续其他工作 | 真实 Web 迁移门禁保持未完成 |
| checkpoint 无法导出 | 优先修复并检查格式与路径 | 本周最高优先级，不能跳过后宣称两阶段 |
| GRPO 全组同分 | 检查解析、题目难度、采样与奖励 | 可用固定样例验证数学实现，真实更新仍须单独通过 |

发生延期时，优先保证“Wiki 两算法更新 + checkpoint 加载 + 真实 Web 单步更新”；缩减样本量和步数，不删除正确性检查。证据正式训练和大规模评测顺延。

## 9. 第一周完成清单

- [ ] 记录源码基线、实际依赖、硬件与模型 revision。
- [ ] 启动参数完整，缺失配置能够提前失败。
- [ ] Wiki 与真实 Web 检索都能通过统一接口返回结果。
- [ ] 数据划分、证据映射与有效样本比例可追踪。
- [ ] 混合 Wiki 数据的 `index` 全局唯一；GRPO 与 Tree 混合 batch 的 `uid` 分组不串题。
- [ ] E0 能产生逐题结果与工具轨迹。
- [ ] GRPO、Tree-GRPO 各完成有效更新与保存。
- [ ] 树分组、共享前缀、observation mask 通过检查。
- [ ] 证据奖励完成离线验证与小步接线；未完成项明确记录。
- [ ] Tree 叶子的金标准证据及实际可见文档 ID 在树内奖励回调一一对应。
- [ ] Wiki actor 可导出、重新加载并完成 Web 更新。
- [ ] 精确续训与仅加载权重的支持范围明确区分。
- [ ] 日志包含物理生成 token、逻辑工具调用、外部请求和 GPU 时间。
- [ ] 有运行说明、初步结果表、失败案例与第二周资源估算。

全部必要门禁通过后，项目可以进入第二周的正式 Wiki 实验。本周成果可表述为“完成基于 Tree-GRPO 的 Wiki/Web 两环境训练工程闭环”；任何性能提升、证据召回改善或成本优势，必须等正式对照结果产生后再写入简历。

## 10. 工单拆分与依赖顺序

| 工单 | 范围 | 依赖 | 完成证据 |
|---|---|---|---|
| W1-01 | 版本固定、环境预检、参数化启动 | GPU/模型资源确认 | 环境清单与预检输出 |
| W1-02 | 统一样本与数据 manifest、全局唯一整数 index | 数据来源确认 | parquet、manifest、跨数据集同原始行号的 uid 分组检查 |
| W1-03 | Wiki/Web 工具适配与错误分类 | API/索引可访问 | 契约测试、真实调用记录 |
| W1-04 | 统一轨迹日志与 E0 评测 | W1-02、W1-03 | 逐题结果、成本汇总 |
| W1-05 | 答案/格式/证据奖励与树内叶子元数据接线 | W1-02、W1-04 | 样本→树→叶子→奖励一一对应的断言结果 |
| W1-06 | GRPO 与 Tree-GRPO 更新验证 | W1-01、W1-04、基础奖励 | 更新指标、mask 检查、树结构 |
| W1-07 | checkpoint 导出、加载与 Web warm start | W1-06、真实 Web 工具 | 父模型记录、新进程更新日志 |
| W1-08 | 汇总评测、运行说明、资源估算 | W1-04 至 W1-07 | 周报告及已通过/未通过清单 |

建议每个工单单独提交；修改底层 rollout 或优势计算时记录相对上游的差异。第一周的短训练统一命名为 smoke，不冒充正式 E1–E4 结果。

## 11. 第二周资源估算方法

第一周记录稳定运行后的 step 时间、每步 prompt 数、新生成 token、逻辑工具调用、真实 API 请求及失败率。按目标有效 step 数估算 GPU 时间：

`GPU 小时 ≈ 稳态秒/step × 目标 step 数 × GPU 数 / 3600`

`API 费用 ≈ 各类实际计费请求量 × 对应单价`

把检索服务占用、模型加载、checkpoint 写入、评测和重试单独列出。训练规模由样本数、epoch、有效 prompt batch 与具体 sampler 共同决定，不能仅凭“12K 数据”推算训练时长；Web 延迟和长轨迹也使小样本线性外推不可靠。报告乐观/基准/保守三档，正式排期采用测量结果。

## 12. 参考底座

- 主仓库：https://github.com/AMAP-ML/Tree-GRPO
- 搜索训练链路：https://github.com/PeterGriffinJin/Search-R1
- 分布式训练框架：https://github.com/verl-project/verl
- Web 工具与数据参考：https://github.com/inclusionAI/ASearcher
- WebDancer / WebWalker 资源：https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent

以上为前期选型来源；具体 API、数据版本与依赖兼容性仍在第一天对选定版本核验。

## 13. 本次修订与验收状态

本次完成源码静态核查与开发文档修订，尚未执行安装、API 调用或 GPU 训练，完成清单保持未勾选。主要修正：复用已有树算法和 Redis 服务；明确两种检索返回格式；纠正 n_agent 与 ts_k 语义；保留原有分段奖励并新增证据项；区分底层 checkpoint 加载和完整训练恢复；将基础验证与正式准确率统计分开。

源码引用行号对应本次读取的快照，后续修改可能移动；定位时同时使用文件名和函数名。此文档放在项目根目录，运行训练与测试时进入 Tree-GRPO-main。
