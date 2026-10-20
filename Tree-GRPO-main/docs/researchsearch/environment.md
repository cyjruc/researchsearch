# ResearchSearch-RL 环境与进度记录

更新时间：2026-10-20（Asia/Shanghai）

## 本地开发机

| 项目 | 已观测值 | 说明 |
|---|---|---|
| 系统 | macOS 15.7.3，Darwin 24.6.0 | 本机代码与文档环境 |
| 芯片 | Apple M4，10 核 | 不是 CUDA 训练设备 |
| 内存 | 16 GB | 训练集群资源尚未提供 |
| Python | 3.14.0，/opt/homebrew/bin/python3 | 与仓库 RL 环境建议版本不一致 |
| 磁盘 | 挂载卷 460 GiB，剩余约 124 GiB | 本机 df 观测值 |
| 工具 | conda、uv、git 可执行；nvidia-smi、nvcc、Ray CLI 不可用 | 仓库目录当前没有 .git 元数据 |
| Python 包 | 当前 Python 未发现 PyTorch、Ray、vLLM、Transformers、Datasets | 不代表目标 GPU 主机状态 |
| CUDA GPU | 不可用 | 无法在本机完成 CUDA 模型生成、优化器更新或 Ray GPU smoke |

## 仓库建议的隔离环境

以下为上游 README 建议，尚未在目标服务器验证安装成功。RL 与检索分别创建环境，避免混装 FAISS 与 vLLM 依赖。

### RL 训练环境

- Python 3.12.9
- PyTorch、torchvision、torchaudio 2.6.0
- vLLM 0.8.5.post1
- 仓库可编辑安装：pip install -e .
- Flash Attention：pip install flash-attn --no-build-isolation
- 依赖成功后记录 python -m pip freeze 与 GPU、驱动、CUDA 版本。

### Retriever 环境

- Python 3.10.13
- PyTorch、torchvision、torchaudio 2.6.0
- Transformers、Datasets、Pyserini
- FAISS GPU 1.7.3
- FastAPI、Uvicorn
- 依赖成功后单独保存 python -m pip freeze。

仓库 README 是版本选择来源，不是本机安装记录。当前 Mac 环境不安装这套 CUDA 专用栈；应在兼容的 Linux GPU 训练机和检索机创建并验证。

## 源码基线

本地 Tree-GRPO-main 没有 Git 元数据，不能报告上游 commit。以下 SHA-256 记录关键文件开发前的基线；正式启用 Git 后以提交记录追踪代码改动。

| 文件 | SHA-256 |
|---|---|
| README.md | f72839c803fdf2d66c9fdc0e09df901f710fd9825176a5783893d5378dbbf587 |
| pyproject.toml | ff22fbfefc7d9e5b6e99813b7459f5bfa76a78b16564e0c25e4d2866e9803ea3 |
| train_multihopqa_grpo.sh | 2c048b6584e0a036854c47fd9edc7e1c0b4c613918f15c337cc2cf607c8f572 |
| train_multihopqa_tree_search.sh | d1dd6b8dda6ffbe7c6c97c72132c56083ea3d2c3ccc213d3cdeb36789807cba0 |

全仓库源码快照（244 个文件，2026-10-20 生成）见 [source-snapshot.md](source-snapshot.md)。

## 启动预检

预检脚本：scripts/researchsearch/preflight.py。它不会打印凭据值。

查看本机信息：

    python3 scripts/researchsearch/preflight.py --report-local

训练启动前，设置以下环境变量。数据默认要求 train.parquet 与 dev.parquet（开发集；正式 test 集锁定）；可分别用 TRAIN_FILE、DEV_FILE 覆盖。

    export DATA_DIR=/path/to/smoke-data
    export MODEL_PATH=/path/to/Qwen2.5-3B
    export LEARNING_RATE=1e-6
    export N_GPUS_PER_NODE=1
    export LOG_DIR=/path/to/researchsearch-logs
    export RAY_ADDRESS=http://127.0.0.1:8265
    export RETRIEVER_URL=http://127.0.0.1:8000/retrieve

先渲染 GRPO 或 Tree-GRPO 的短任务命令，不会提交 Ray 作业：

    ALGORITHM=grpo bash scripts/researchsearch/launch_smoke.sh --dry-run
    ALGORITHM=tree bash scripts/researchsearch/launch_smoke.sh --dry-run

目标机满足依赖、GPU 与 Ray 检查后才提交：

    ALGORITHM=grpo bash scripts/researchsearch/launch_smoke.sh --submit
    ALGORITHM=tree bash scripts/researchsearch/launch_smoke.sh --submit

实际提交会检查 CUDA 包、可见 GPU 数、Ray Jobs endpoint，并提交一个轻量 Ray worker probe，核验 runtime_env 变量、GPU、模型/数据路径和 worker 到检索服务的 TCP 连通性。启动搜索服务后，可单独发一条真实搜索请求（会消耗服务配额）：

    python3 scripts/researchsearch/preflight.py --probe-retriever

模型短生成和一次 AdamW 更新可在目标机单独检查：

    python3 scripts/researchsearch/preflight.py --model-smoke

Web 搜索服务使用单独的 retriever 环境。先启动 Redis，设置 BRIGHTDATA_API_KEY 环境变量，再运行 bash scripts/researchsearch/launch_search_service.sh；凭据不会作为命令行参数传递。服务就绪后，可通过 PROBE_RETRIEVER=1 配合 --submit 的启动预检发送一次真实搜索请求，并查看返回状态与结果数。不要把密钥写入 launcher、日志或配置文件。

## Day 1 状态

- [x] 本机系统、芯片、内存、磁盘与 Python 版本已记录。
- [x] 核实本机不是 CUDA 训练机；未伪报 GPU、Ray 或 API smoke 成功。
- [x] 检查源码目录无 Git 元数据，记录关键文件哈希。
- [x] 新增参数化 preflight 和 GRPO/Tree smoke 启动器。
- [x] 新增从环境变量读取搜索服务凭据的启动器；改造原 Bing 启动入口以委托新脚本。
- [ ] 在目标 Linux GPU 主机安装并锁定两个环境。
- [ ] 核验 Ray runtime_env 在 worker 上分发代码及模型、数据路径。
- [ ] 在目标 GPU 上完成 tokenizer、生成和一次 optimizer step。
- [ ] 配置检索/Web 服务后发起并记录一次真实搜索请求。

目标 GPU 主机、模型与数据挂载路径、Ray 地址及搜索服务凭据当前尚未提供，因此后四项保持待执行。

## Day 2 状态（数据层，不依赖 GPU / 检索服务）

- [x] 新增统一数据构建模块 `researchsearch/data/build.py`：全局唯一整数 `index`、规范化去重、train/dev 泄漏检测、`sample_id`、证据字段与 `manifest`。
- [x] 提供 `python -m researchsearch.data.build` CLI，支持本地 JSONL（本机 smoke）与 FlashRAG 目录（目标机，`datasets` 延迟导入）。
- [x] 单元测试 `tests/researchsearch/test_data_build.py`（11 项，覆盖 T01 分组 ID、去重、泄漏、证据字段与 prompt 隔离），全部通过。
- [x] 编写检索契约 `docs/researchsearch/retrieval-contract.md`，如实区分当前实现与方案缺口（document 无稳定 id、Web 缓存键/失败结构、客户端超时）。
- [ ] 真实语料/索引就绪后：生成 128/64 条 smoke parquet + manifest，补齐 `doc_id` 与证据映射报告。

证据映射、真实 parquet 与 `/retrieve` 契约测试仍依赖语料与索引，属 Day 2 卡住部分。

## 进度汇总（截至 2026-10-20，非阻塞收尾）

以下为 Day 3–4 期间完成、且不依赖 GPU / 真实语料 / 检索服务的部分，全部已通过本地单元测试。

### 本轮完成

- [x] **T04 分段奖励**：新增 `tests/researchsearch/test_reward_format.py`（14 项），锁定 qa_em / qa_f1 四种“正确/错误 × 合法/不合法”语义，并如实记录两处上游不一致（F1“错误+不合法”返回 0、EM 返回 0.1；`extract_solution` 需 ≥2 个 `<answer>`）。
- [x] **T05 证据奖励 R_evidence**：新增纯函数 `verl/utils/reward_score/evidence.py`（文档级覆盖 = 去重命中 gold 数 / |gold|），配套 `tests/researchsearch/test_evidence.py`（8 项：全覆盖 / 部分 / 重复不加分 / 空 gold 不除零 / 非 mapped 关闭 / partial 单独报告 / retrieval_hit 不计分子 / 去重保序）。
- [x] **T07 优势计算**：新增 `tests/researchsearch/test_core_algos.py`（9 项），锁定 grpo 按 index 标准化、tree 默认 = 两次独立标准化相加、tree_2norm = 顺序双重标准化、per-token 池化等语义。
- [x] **T06 前置静态分析**：新增 `docs/researchsearch/loss-mask.md`，记录 info_mask → `_create_loss_mask` → loss_mask → response_mask → agg_loss 全链路；结论：loss mask 生效前提是 `state_masking=true`（train 脚本与 smoke 启动器均已覆盖）。
- [x] **数据层迁移收尾**：`data_process_multihop.sh` / `data_process_singlehop.sh` 改为 `python -m researchsearch.data.build`；旧 `qa_search_train_merge.py` / `qa_search_test_merge.py` 标注 deprecated 并指向新入口；下游默认验证集统一为 `dev.parquet`（6 个 train 脚本 + `launch_smoke.sh` + `preflight.py` + `ray_worker_probe.py`）。
- [x] **客户端检索超时/重试/错误分类**（方案 §2.1 阻塞 6 / §4.3）：新增 `search_r1/search/retrieval_client.py`（`post_retrieve` 单次 30s、最多重试 2 次、指数退避、区分 auth/rate_limit/server/network/invalid），`generation.py` / `generation_ts.py` 的 `_batch_search` 改走该客户端；配套 `tests/researchsearch/test_retrieval_client.py`（16 项）。
- [x] **全仓库源码快照**（Day 1 任务 2 补全）：新增 `docs/researchsearch/source-snapshot.md`，记录 244 个源码文件 SHA-256（排除 .pytest_cache / __pycache__ / 图片二进制）。
- [x] **T06 loss mask 单元测试**：新增 `tests/researchsearch/test_loss_mask.py`（6 项），加载真实 `_info_masked_concatenate_with_padding` 与 `TensorHelper.create_attention_mask`，锁定 information → pad_id → mask 0 的构造、`loss_mask = info_mask[:, -response_length:]` 切片与 `state_masking` 开关语义。
- [x] **树内证据奖励接线预备**（方案 Day 5 任务 6 / Day 6 任务 1 无卡部分）：新增 `docs/researchsearch/evidence-wiring.md`，记录字段传递全链路、三个丢失点（pop 只留 3 字段 / `_passages2string` 丢 doc_id / RewardManager 不读证据）与 Day 6 接线蓝图；新增 `evidence.extract_doc_ids_from_retrieval` 纯函数（旁路提取 Wiki `document.id`）；配套 `tests/researchsearch/test_tree_evidence_wiring.py`（6 项，锁定叶子顺序与 `np.repeat` 行序对齐、doc_id 提取、两题不串题）。核心训练路径改动待真实 rollout 验证。
- [x] **T07 优势计算路由测试**（方案 Day 5 任务 3）：新增 `tests/researchsearch/test_adv_routing.py`（4 项），加载真实 `ray_trainer_ts.compute_advantage`，锁定 `tree` = 两次 `compute_grpo_outcome_advantage` 相加、`tree_2norm` = 一次 `compute_tree_grpo_outcome_advantage`，二者不同不能混用。
- [x] **checkpoint 导出参数化**（方案 §6 Day 6 任务 2 / §2.1 阻塞 4）：新增 `scripts/researchsearch/export_ckpt.sh`，通过自身目录定位 `model_merger.py`，参数化 `LOCAL_DIR` / `TARGET_DIR` / `HF_MODEL_PATH` / `BACKEND`，支持 `--dry-run`；旧 `convert_checkpoint_from_verl_to_hf.sh` 标注 deprecated 并指向新脚本；配套 `tests/researchsearch/test_export_ckpt.py`（5 项，缺参数提前失败、dry-run 路径解析、hf_model_path 仅按需透传）。真实合并/加载与确定性推理一致性检查待真实 checkpoint（GPU 阶段）。
- [x] **Day 1–7 一致性审计修复**（对齐方案 §4.1 / 架构 §4）：① 修正 `build.py` 的 `--dev-flashrag` 错误消费 FlashRAG `test` split（违背「正式测试集锁定」），改为 `_DEV_FLASHRAG_SPLIT="dev"`，并补 3 项测试锁定 dev 优先 dev split、无 dev 回退 train、绝不落 test；② 新增 `evidence.compute_recall_at_k` 纯函数（架构 §4 检索指标：按检索顺序去重后取前 K）+ 5 项测试；③ 补 `test_loss_mask.py` tree 版（`generation_ts`，含 `turns_mask`/`turns`）掩码构造 4 项测试，锁定 information→pad→mask0 与 response content 段=当前 turn 值语义。

### 测试状态

91 项全部通过：`data_build` 14（T01 + FlashRAG split 锁定）+ `reward_format` 14（T04）+ `evidence` 13（T05 + Recall@K）+ `core_algos` 9（T07）+ `retrieval_client` 16（T03 客户端）+ `loss_mask` 10（T06 chain + tree）+ `adv_routing` 4（T07 路由）+ `tree_evidence_wiring` 6（Day 6 预备）+ `export_ckpt` 5（T09 脚本）。

### 门禁对照

| 门禁 | 状态 | 说明 |
|---|---|---|
| T01 分组/索引 | ✅ | build.py 全局唯一 index + dev split 锁定（不消费 test），data_build 覆盖 |
| T02 证据映射 | ⛔ | 依赖真实 FlashRAG 语料/索引 |
| T03 检索契约 | 🟡 | 客户端超时/重试/错误分类已完成（16 项）；服务端批量顺序、doc_id 仍需真实索引 |
| T04 分段奖励 | ✅ | 14 项测试锁定语义 |
| T05 证据奖励 | ✅ | evidence.py + 13 项测试（R_evidence 8 + Recall@K 5）+ doc_id 提取纯函数 + 接线预备测试 |
| T06 loss mask | 🟡 | 静态分析 + 单元测试完成（10 项：chain 6 + tree 4），运行时断言待真实 rollout |
| T07 优势计算 | ✅ | 9 项 core_algos + 4 项路由测试锁定 |
| T08 权重更新 | ⛔ | 需真实 GPU rollout 与 optimizer step |
| T09 checkpoint | 🟡 | 导出脚本已参数化（5 项）；真实合并/加载/下一阶段更新待真实 checkpoint |
| T10 预算 | ⛔ | 需真实 rollout 统计缓存/重试/逻辑调用 |

### 剩余阻塞项

- [ ] 目标 Linux GPU 主机安装并核验两个 conda 环境（RL / Retriever），完成 Ray runtime_env、tokenizer 生成 + 一次 optimizer step。
- [ ] 真实 FlashRAG 语料/索引就绪后：生成 128/64 条 smoke parquet + manifest，补齐 `doc_id` 与证据映射报告。
- [ ] 检索服务就绪后：执行 `/retrieve` 契约测试。
- [ ] 证据奖励接入树内回调（Day 6）：接线蓝图与预备测试已交付（`docs/researchsearch/evidence-wiring.md` + `test_tree_evidence_wiring.py`），核心路径改动（pop 增字段 → 逐叶 `visible_doc_ids` → RewardManager 证据项）待真实 rollout 验证。
- [ ] loss mask 运行时断言（真实 rollout）。
