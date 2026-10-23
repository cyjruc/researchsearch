# ResearchSearch-RL 第 2 周 Wiki 实验报告

版本：v0.2（数据就绪；GPU 训练 / 评测项待回填）· 2026-10-23
对应方案：《第 2 周 Wiki 实验开发方案》§7、§8、§10
对应架构：《项目架构与实验设计》§4（评测）、§5（第 2 周：Wiki 实验）

> 本文件为周报骨架。标「待回填」的数值与结论不得在真实数据产出前填写；
> 任何性能提升 / 证据召回改善 / 成本优势，须待正式对照结果产出后写入（方案 §10）。

## 1. 本周目标与结论速览

- 目标：完成 12K Wiki 数据与证据映射、证据奖励接线验证、E1/E2/E3 三个 Wiki
  checkpoint、EM / Recall@K 与奖励消融，为第三周 Web 迁移留下输入。
- 成功标准：三个 Wiki checkpoint 可复核、消融结论可追溯到逐题结果（不要求复现
  上游 31.8 或简历中的 36.2 等数值）。
- 结论速览：**待回填**（三个 checkpoint 的 EM / Recall@K / 消融结果产出后填写）。

## 2. 完成清单（对照方案 §10）

- [x] 12K Wiki 数据（NQ 2K + HotpotQA 4K + 2Wiki 3K + MuSiQue 3K）与独立 dev 生成，manifest 含抽样种子 —— train 11997 / dev 511，train/dev 泄漏 0（§4.2）
- [x] 证据映射（`researchsearch/data/evidence_map.py`：四态判定 / 歧义不猜测 / 语料缺失不误判 / 去重 / Unicode NFD↔NFC 归一化回退）；真实语料可映射率 / partial / 歧义 / 语料缺失报告已产出（§4.2）
- [x] 树内证据奖励接线代码落地（`extra_info`→`gen_batch`→逐叶 `visible_doc_ids`→`RewardManager`，Tier2）+ 真实 doc_id 提取修复（`extract_doc_ids_from_retrieval` 只认 `wiki:` 前缀、段落级行号回退 title，离线契约验证 2000/2000 命中 gold）；两题不串题、真实覆盖率非零 —— 待 GPU 真实 rollout 验证
- [x] 「截断后可见」判定纯函数 + 单元测试（`evidence.split_doc_ids_by_truncation`，9 项）
- [x] E1/E2/E3 配置统一（`launch_wiki.sh` + `experiment-config.md`），算法相关差异与预算停止规则已记录
- [ ] 小规模检查通过，正式超参冻结 —— 待 GPU
- [ ] E1、E2、E3 三个 Wiki checkpoint 训练完成、可导出、可重载 —— 待 GPU
- [x] 评测评分器 + 消融框架（`eval_metrics.py`：EM/F1 判定、宏平均、Multi-Hop Avg EM、bootstrap CI、`pairwise_ablation`；27 项测试）；真实评测数值 —— 待 GPU
- [x] 评测编排入口（`eval_runner.py` + `scripts/researchsearch/evaluate.py`：逐题预测 schema 校验、EM/F1/Recall@K/可映射比例报告、两实验消融 CLI；16 项测试）；真实评测数值 —— 待 GPU
- [x] 成本统计代码/口径/训练接线（`cost_stats.py` + `cost-stats.md` + rollout/trainer 埋点：物理 token / GPU 秒 / checkpoint 体积随训练 metrics 上报）；真实填数 —— 待 GPU+检索服务
- [x] Multi-Hop Avg EM 测试集 manifest 骨架（`eval_manifest.py` + `scripts/researchsearch/gen_eval_manifest.py`：四数据集完整性强制、只读哈希、篡改校验；11 项测试）；真实题目 ID / 数据版本预注册 —— 待真实评测集核对
- [ ] week2-report.md 完成（本文件），第三周 Web 迁移输入齐备 —— 待回填

## 3. 门禁对照

### 3.1 本周 Wiki 实验门禁（E01–E07）

| 编号 | 验证对象 | 状态 | 说明 |
|---|---|---|---|
| E01 | 12K 数据与证据映射 | ✅ | 12K train/dev 已生成（泄漏 0、index 全局唯一）；映射率已报告（§4.2）；gold 文档与冻结语料一致性已核对（wiki-18 覆盖 ~57%，2Wiki 缺口如实标注） |
| E02 | 证据接线验证与 doc_id 打通 | 🟡 | 接线代码 + 截断判定已交付；`extract_doc_ids_from_retrieval` 段落级行号 bug 已修复 + 离线契约验证（2000/2000 命中 gold）；真实覆盖率非零仍待 GPU 检索服务 |
| E03 | E1/E2/E3 配置统一 | ✅ | `launch_wiki.sh` + `experiment-config.md`（`COMMON_OVERRIDES` 单一来源 + 算法差异分叉） |
| E04 | 小规模稳定性 | ⛔ | 待 GPU（无 NaN/Inf、非零更新） |
| E05 | 三 Wiki checkpoint | ⛔ | 待 GPU（训练 + `export_ckpt.sh` 导出 + 重载） |
| E06 | 评测口径 | 🟡 | `eval_metrics.py`（EM 独立于训练奖励）+ `compute_recall_at_k`（与证据奖励集合分离）已交付；真实数值待 GPU |
| E07 | 消融可复现 | 🟡 | `pairwise_ablation` 框架已交付（逐题可追溯）；真实消融数值待 GPU |

### 3.2 第 1 周遗留门禁（本周须转 ✅）

| 编号 | 状态 | 说明 |
|---|---|---|
| T02 数据映射 + 真实 doc_id 打通 | 🟡 | 数据映射已完成（12K + 映射报告 + doc_id 提取修复）；真实覆盖率非零仍待 GPU 检索服务 |
| T03 真实索引契约测试 | 🟡 | 格式层契约已交付（fake retriever）；语料 / 索引 / 模型文件已就绪，但 GPU 不可用，真实检索服务待开机 |
| T05 接线验证 + alpha 消融 | 🟡 | 纯函数 + 接线 + 截断判定已交付；真实覆盖率与 alpha 消融数值待 GPU |
| T10 真实预算统计 | 🟡 | 六口径代码/口径已交付（`cost-stats.md`）；真实填数待 GPU+检索服务 |

## 4. 运行说明

### 4.1 数据准备（12K + dev + manifest）
```bash
# 真实 FlashRAG 语料就绪后：build.py 生成 train.parquet / dev.parquet / manifest.json
# 指定 split 不存在时直接报错，不自动回退（保持 train/dev 隔离）。
```

### 4.2 证据映射报告

2026-10-23 用 wiki-18 语料 + `title_docid_map.json`（3,232,908 唯一 title）产出 train/dev
两份报告（`evidence_mapping_report_{train,dev}.json`）。mapped 率：

| 数据源 | train | dev |
|---|---|---|
| HotpotQA | 80.3% | 80.7% |
| 2WikiMultiHopQA | 41.7% | 51.0% |
| MuSiQue | 70.4% | 72.1% |
| NQ | 无文档级证据（not_available） | 同左 |

关键发现：

- **Unicode 归一化**：wiki-18 语料 title 以 NFD 分解形式存储，FlashRAG 支持事实以
  NFC 预组合形式给出，字符串不等价导致变音符号 title 被误判为语料缺失。已在
  `evidence_map._resolve_title` 加 NFD/NFC 归一化回退（原样优先），修复后 dev mapped
  率 HotpotQA 77.1→80.7、2Wiki 44.6→51.0、MuSiQue 69.1→72.1。
- **语料内在缺口**：wiki-18 只覆盖约 57% 英文维基（21,015,324 段落 / 3,232,908 唯一
  title）。2Wiki 的 mapped 率仍低（~51%）主要因语料缺电影导演 / 老片等条目（如
  Phil Rosen、Ray Enright、Éric Rohmer、Owain Glyndŵr），归一化救不了语料本身缺的
  title。此缺口非搜索策略失败，须在报告与评测中如实标注（架构 §2.1）。
- **歧义为 0**：title 级归并正确，同名 title 多 chunk 已在映射表构建阶段归并。
- **人工核对**：每源抽样 7 条（nq 无支持事实不纳入），train/dev 各 21 条有支持事实
  样本（hotpotqa/2wiki/musique 各 7）。

### 4.3 检索服务
```bash
# Wiki：FAISS/E5 /retrieve（retriever env）
# 评测预算与训练预算分开固定；Recall@10 前置要求检索预算 ≥10 不同文档（方案 §3.3）。
```

### 4.4 配置统一与启动
```bash
bash scripts/researchsearch/launch_wiki.sh e1 --dry-run   # 或 e2 / e3；--submit 提交
# 三实验只在「算法相关差异」处分叉；其余 COMMON_OVERRIDES 逐项一致。
```

### 4.5 评测与消融
```bash
# 单实验评测（Wiki dev 或四项测试集）：EM/F1/Recall@K/可映射比例 + bootstrap CI
python scripts/researchsearch/evaluate.py --predictions preds.json --out report.json

# 奖励消融（E2 vs E1、E3 vs E2），逐题可追溯
python scripts/researchsearch/evaluate.py --baseline e1.json --comparison e2.json \
    --metric em --label-a E1 --label-b E2 --out ablation.json
```
逐题预测 JSON 的 schema 见 `verl/utils/reward_score/eval_runner.py` 顶部注释。
评测核心由 `eval_metrics`（`em_match`/`f1_match`/`mean_scores`/`multi_hop_avg_em`/
`bootstrap_ci`/`pairwise_ablation`）与 `evidence.compute_recall_at_k` 提供；
`eval_runner` 锁定编排口径：EM 独立于训练奖励（不含格式分段）、Recall@K 读
`retrieved_doc_ids` 与证据集合 `visible_doc_ids` 分离、Multi-Hop Avg EM 四项等权
缺项如实标注不静默。

### 4.6 输出定位
数据与映射报告已产出：`/root/autodl-tmp/data/wiki12k/`（`train.parquet` / `dev.parquet` /
`manifest.json` / `evidence_mapping_report_{train,dev}.json`）。checkpoint 目录、HF 导出、
逐题评测 JSON、消融对照表路径待 GPU 训练后回填。

## 5. 结果表（待回填）

### 5.1 EM / Recall@K / 证据可映射比例

| 实验 | Wiki dev EM | Recall@5 | Recall@10 | 证据可映射比例 |
|---|---|---|---|---|
| E1 GRPO | 待回填 | 待回填 | 待回填 | 待回填 |
| E2 Tree | 待回填 | 待回填 | 待回填 | 待回填 |
| E3 Tree+证据 | 待回填 | 待回填 | 待回填 | 待回填 |

口径：EM 独立于训练奖励（`eval_metrics.em_match`）；Recall@K 按检索返回顺序去重取前 K
（`evidence.compute_recall_at_k`），与证据奖励的「模型可见集合」分开留档。

### 5.2 奖励消融

| 对照 | 结论 | delta（题级均值） | 逐题追溯 |
|---|---|---|---|
| E2 vs E1（树收益） | 待回填 | 待回填 | `pairwise_ablation(..., label_a="E1", label_b="E2")` |
| E3 vs E2（证据收益） | 待回填 | 待回填 | `pairwise_ablation(..., label_a="E2", label_b="E3")` |

消融结论须可追溯到逐题结果（`per_question` 字段）；未提升的指标如实报告。

### 5.3 成本统计

| 口径 | E1 | E2 | E3 |
|---|---|---|---|
| 物理新生成 token | 待回填 | 待回填 | 待回填 |
| 逻辑搜索次数 | 待回填 | 待回填 | 待回填 |
| 实际外部请求（含重试） | 待回填 | 待回填 | 待回填 |
| GPU 时间（小时） | 待回填 | 待回填 | 待回填 |

口径与填数清单见 [cost-stats.md](./cost-stats.md)；共享前缀不计入物理新生成 token。

## 6. 失败案例（待回填）

真实评测 / 训练后填写：答案错误、证据召回失败、无效调用、超时、达到预算上限的典型
逐题案例，附完整工具轨迹。

## 7. 第三周任务与资源估算（待回填）

- 三个 Wiki checkpoint（E1/E2/E3 的 HF 导出）+ 测试集 manifest 是第三周 Web 迁移输入。
- 资源估算用本周正式训练实测的稳态秒/step 外推：
  `GPU 小时 ≈ 稳态秒/step × 目标 step 数 × GPU 数 / 3600`（方案 §11）。
  报告乐观/基准/保守三档；检索服务占用、模型加载、checkpoint 写入、评测、重试单独列出。

## 8. 交付检查表

- [ ] 12K Wiki 数据 + dev + manifest（抽样种子）
- [ ] 证据映射小样本报告
- [ ] 树内证据接线验证（两题不串题、真实覆盖率非零）
- [ ] 截断后可见判定已交付（纯函数 + 测试）
- [ ] E1/E2/E3 配置统一（已交付）
- [ ] 小规模检查 + 超参冻结
- [ ] 三个 Wiki checkpoint（训练 / 导出 / 重载）
- [ ] EM / Recall@K / 证据可映射比例评测表
- [ ] 奖励消融（E2 vs E1、E3 vs E2）逐题可追溯
- [x] 测试集 manifest 骨架（真实预注册待评测集核对）
- [x] 成本统计接线齐全（物理 token / 逻辑搜索 / 外部请求 / GPU 时间；真实数值待 GPU）
- [ ] week2-report.md 完成

## 9. 待回填清单

| 项 | 依赖 | 回填位置 |
|---|---|---|
| 12K 数据与证据映射数值 | ✅ 已完成 | §2、§4.2 |
| 两题不串题 + 真实覆盖率非零 | GPU + 语料 + 检索服务 | §3.1 E02 |
| 小规模稳定性、正式超参冻结 | GPU | §3.1 E04、§4.4 |
| 三 checkpoint 训练/导出/重载 | GPU | §3.1 E05 |
| EM/Recall@K/证据可映射比例 | GPU + 语料 | §5.1 |
| 消融数值（E2 vs E1、E3 vs E2） | GPU + 语料 | §5.2 |
| 成本统计真实数值 | GPU + 检索服务 | §5.3、[cost-stats.md](./cost-stats.md) |
| 测试集 manifest 预注册 | 真实评测集核对 | §2、§8 |
| 失败案例、资源估算 | 以上数据 | §6、§7 |
