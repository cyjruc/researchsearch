# ResearchSearch-RL 环境与进度记录

更新时间：2026-10-22（Asia/Shanghai）

## 本地开发机

| 项目 | 已观测值 | 说明 |
|---|---|---|
| 系统 | macOS 15.7.3，Darwin 24.6.0 | 本机代码与文档环境 |
| 芯片 | Apple M4，10 核 | 不是 CUDA 训练设备 |
| 内存 | 16 GB | 本机内存（非训练机）；GPU 训练在 AutoDL A800 主机，见下 |
| Python | 3.14.0，/opt/homebrew/bin/python3 | 与仓库 RL 环境建议版本不一致 |
| 磁盘 | 挂载卷 460 GiB，剩余约 124 GiB | 本机 df 观测值 |
| 工具 | conda、uv、git 可执行；nvidia-smi、nvcc、Ray CLI 不可用 | 仓库已 git 化：root /Users/ct/code/算法/researchsearch，remote git@github.com:cyjruc/researchsearch.git，HEAD 744dda7 |
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

## 目标 GPU 主机实测（AutoDL）

目标机为 AutoDL 按需 GPU 主机，`ssh -p 11824 root@connect.nma1.seetacloud.com`。以下均为真实观测值（2026-10-22）。

### 硬件

| 项目 | 已观测值 |
|---|---|
| GPU | NVIDIA A800 80GB PCIe（81920 MiB），driver 590.48.01 |
| CPU | 112 核 |
| 内存 | 991 GiB |
| 磁盘 | 系统盘 30G（剩约 13G）；数据盘 /root/autodl-tmp 80G（剩约 75G） |

模型、代码、日志统一放数据盘 `/root/autodl-tmp`，不占用 30G 系统盘。

### RL 环境（treegrpo，实测）

| 包 | 实测版本 |
|---|---|
| Python | 3.12.9 |
| torch | 2.6.0（CUDA 12.4，`torch.cuda.is_available()`=True） |
| vllm | 0.8.5.post1 |
| flash-attn | 2.7.4.post1 |
| ray | 2.59.0 |
| swanlab | 0.10.2 |
| transformers | 5.20.1 |
| datasets | 5.1.0 |
| verl | 0.1（editable，指向 /root/autodl-tmp/researchsearch/Tree-GRPO-main @ 744dda7） |

### Retriever 环境（retriever，实测）

| 包 | 实测版本 |
|---|---|
| Python | 3.10.13 |
| torch | 2.6.0 |
| transformers | 5.20.1 |
| datasets | 5.1.0 |
| pyserini | 1.2.0 |
| fastapi | 0.149.0 |
| uvicorn | 0.54.0 |
| numpy | 1.26.4 |
| faiss-gpu | ❌ 1.7.3 安装失败（该版本不存在于 PyPI） |

两个环境的 `pip freeze` 已记录到远程 `/root/autodl-tmp/researchsearch/env-lock/{treegrpo_freeze.txt,retriever_freeze.txt}`。

### 三段 smoke（真实 GPU，Qwen2.5-3B）

模型 `Qwen2.5-3B`（5.8G）已下载到 `/root/autodl-tmp/models/Qwen2.5-3B`。在真实 GPU 上逐段验证：

1. **tokenizer**：加载与 roundtrip 通过。
2. **短生成**：通过，示例输出 "J.K. Rowling is the author of Harry Potter."
3. **一次 AdamW optimizer step**：通过，loss 2.3828，参数张量已更新。

对应 Day 1 任务 6 的「在目标 GPU 上完成 tokenizer、生成和一次 optimizer step」已真实通过。

### 三个发现

1. **faiss-gpu 1.7.3 是版本笔误**：PyPI 只有 1.7.1.post3 / 1.7.2 / 1.14.3 / 1.15.x，无 1.7.3。属计划/README 笔误，retriever 阶段需改用可用版本；不阻塞当前 RL。
2. **huggingface_hub 2.2.0 的 Xet 存储后端匿名下载 401**：Qwen2.5-3B 走 Xet 存储，默认后端匿名访问被拒。修复：下载前 `export HF_HUB_DISABLE_XET=1`。
3. **代码同步已确认一致**：verl 可编辑安装指向 `/root/autodl-tmp/researchsearch/Tree-GRPO-main`，该目录即 git clone（remote https://github.com/cyjruc/researchsearch.git，HEAD 744dda7 init2），与本机及 GitHub 完全一致；site-packages 中 `compute_recall_at_k`、`_DEV_FLASHRAG_SPLIT` 均已存在，无需额外 push 或重装。

## 源码基线

仓库已 git 化：root `/Users/ct/code/算法/researchsearch`，remote `git@github.com:cyjruc/researchsearch.git`，HEAD `744dda7`（init2）。代码改动以提交记录追踪。下表 SHA-256 是早期（目录尚无 .git 时）关键文件的开发前基线，仅作历史留档，不代表当前内容。

| 文件 | SHA-256 |
|---|---|
| README.md | f72839c803fdf2d66c9fdc0e09df901f710fd9825176a5783893d5378dbbf587 |
| pyproject.toml | ff22fbfefc7d9e5b6e99813b7459f5bfa76a78b16564e0c25e4d2866e9803ea3 |
| train_multihopqa_grpo.sh | 2c048b6584e0a036854c47fd9edc7e1c0b4c613918f15c337cc2cf607c8f572 |
| train_multihopqa_tree_search.sh | d1dd6b8dda6ffbe7c6c97c72132c56083ea3d2c3ccc213d3cdeb36789807cba0 |

全仓库源码快照（256 个文件，2026-10-22 重算，可复现脚本 `scripts/researchsearch/gen_source_snapshot.py`）见 [source-snapshot.md](source-snapshot.md)。

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
- [x] 在目标 Linux GPU 主机安装并锁定两个环境。
- [ ] 核验 Ray runtime_env 在 worker 上分发代码及模型、数据路径。
- [x] 在目标 GPU 上完成 tokenizer、生成和一次 optimizer step。
- [ ] 配置检索/Web 服务后发起并记录一次真实搜索请求。

目标 GPU 主机（AutoDL A800）已就位，两个 conda 环境已安装、三段 smoke（tokenizer / 生成 / optimizer step）已在真实 GPU 通过。仍待执行的是 Ray runtime_env 探针核验与检索服务配置后的真实搜索请求。

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
- [x] **树内证据奖励接线落地**（方案 Day 6 任务 1 无卡部分）：接线代码已落地——`ray_trainer_ts.py` 条件性把 `extra_info` 带入 `gen_batch`、`generation_ts.py` 旁路累积逐叶 `visible_doc_ids` 并在 `reward_fn` 前写入 `final_output`、`RewardManager` 新增 `evidence_alpha` 证据项；配套 `evidence.extract_doc_ids_from_retrieval` / `compute_evidence_reward` / `combine_score_with_evidence` 纯函数与 `tests/researchsearch/test_tree_evidence_wiring.py`（12 项，锁定叶子顺序对齐、doc_id 提取、alpha 语义、`visible_doc_ids` 隔离、端到端不串题）。真实 rollout 逐叶验证与「截断后可见性判定」待 GPU+语料（见 `docs/researchsearch/evidence-wiring.md` §6）。
- [x] **T07 优势计算路由测试**（方案 Day 5 任务 3）：新增 `tests/researchsearch/test_adv_routing.py`（4 项），加载真实 `ray_trainer_ts.compute_advantage`，锁定 `tree` = 两次 `compute_grpo_outcome_advantage` 相加、`tree_2norm` = 一次 `compute_tree_grpo_outcome_advantage`，二者不同不能混用。
- [x] **checkpoint 导出参数化**（方案 §6 Day 6 任务 2 / §2.1 阻塞 4）：新增 `scripts/researchsearch/export_ckpt.sh`，通过自身目录定位 `model_merger.py`，参数化 `LOCAL_DIR` / `TARGET_DIR` / `HF_MODEL_PATH` / `BACKEND`，支持 `--dry-run`；旧 `convert_checkpoint_from_verl_to_hf.sh` 标注 deprecated 并指向新脚本；配套 `tests/researchsearch/test_export_ckpt.py`（5 项，缺参数提前失败、dry-run 路径解析、hf_model_path 仅按需透传）。真实合并/加载与确定性推理一致性检查待真实 checkpoint（GPU 阶段）。
- [x] **Day 1–7 一致性审计修复**（对齐方案 §4.1 / 架构 §4）：① 修正 `build.py` 的 `--dev-flashrag` 错误消费 FlashRAG `test` split（违背「正式测试集锁定」），改为 `_DEV_FLASHRAG_SPLIT="dev"`，并补 3 项测试锁定 dev 优先 dev split、无 dev 回退 train、绝不落 test；② 新增 `evidence.compute_recall_at_k` 纯函数（架构 §4 检索指标：按检索顺序去重后取前 K）+ 5 项测试；③ 补 `test_loss_mask.py` tree 版（`generation_ts`，含 `turns_mask`/`turns`）掩码构造 4 项测试，锁定 information→pad→mask0 与 response content 段=当前 turn 值语义。
- [x] **树结构验证（Item 2）**：`TreeNode.to_dict` / `export_tree_roots` 树导出由 `tests/researchsearch/test_tree_export.py`（8 项）锁定，共享前缀复用与叶子顺序对齐由 `tests/researchsearch/test_tree_shared_prefix.py`（4 项）锁定。遗留子项「共享前缀 log_prob 对齐」澄清：`TreeNode.log_prob_node` / `log_prob_list` 当前未填充（`generation_ts.py` 建节点时不传这两个参数，保持默认 `0.0` / `[]`），但训练不依赖它们——`ray_trainer_ts.py:887` 对拼好共享前缀的完整轨迹逐 token `compute_log_prob` 得到 `old_log_probs`，论文重要性比率同样是逐 token 计算，故无需先把推理 log_prob 归约成节点分数；`_get_action_log_probs` 算出的 `log_probs_step` 仅是生成侧统计，不进入节点字段。
- [x] **Web 缓存命名空间 + 预算口径统计 + 结构化失败状态与旁路元数据**（方案 Day 6 任务 3 / 门禁 T10 无卡部分）：`bing_search_server_redis.py` 的 `OnlineSearchEngine` 缓存键从裸 `query` 改为 `{provider}:{zone}:{cache_version}:{max_results}:{result_length}:{query}`（`_cache_key`，换 provider/zone/版本/topk 后旧缓存不命中）；新增 `get_stats()` 三口径分开统计——`logical_queries`（Agent 逻辑搜索，每个 query 一次）/`cache_hits`（Redis 命中）/`api_requests`（真实 HTTP 请求，含重试），`execute_with_retry` 达上限终止；`execute` 改为生成 `SearchOutcome`（status/text/items/error_code），`get_last_outcome()` 暴露结构化失败分类（success/not_found/rate_limit/network/server/invalid/auth/empty_query）与 `items` 旁路（rank/title/url/description），鉴权错误立即失败不重试，模型可见返回形状不变。配套 `tests/researchsearch/test_web_cache_budget.py`（19 项，mock redis/fastapi/langid/uvicorn，fixture 锁定命名空间隔离、三口径与失败分类）。
- [x] **源码快照可复现化（B-8）**：新增 `scripts/researchsearch/gen_source_snapshot.py`，对 Tree-GRPO-main 下排除 `.git`/`.pytest_cache`/`__pycache__`/`.DS_Store`/二进制图片的所有文件计算 SHA-256 并写 `docs/researchsearch/source-snapshot.md`；本次重算 256 个文件（2026-10-22），后续任何改动后一条命令即可重算，取代此前一次性命令。
- [x] **第 1 周报告框架（B-7）**：新增 `docs/researchsearch/week1-report.md`，含目标/结论速览、对照方案 §9 的 14 项完成清单、T01–T10 门禁对照、从空环境到 smoke 结果的运行说明、结果表/失败案例占位、第二周任务与三档资源估算方法、交付检查表与待回填清单；GPU/真实服务数据回来后填实结果表（§5）、失败案例（§6）与资源估算数值（§7.2），不预填任何提升数值。
- [x] **真实 GPU 三段 smoke + 结果回填（B-7 后半）**：AutoDL A800 单卡跑通 E0（val_only 逐题导出 `e0_3.json`，4 turn 全 invalid / 0 搜索 / ~74s）、GRPO（2 步，critic 0.600，finish_ratio 0.875，76.6s/step，MFU 0.120，验证 0.0）、Tree（M2/N2/L1/K3，1 步，critic 0.254，pg_loss 0.129，grad_norm 2.085，104.8s/step，MFU 0.234，最终验证 1.0），训练闭环（rollout→优势→更新→checkpoint 保存）验证通过，T06/T08 转 ✅。关键资源发现：单卡 NO_SHARD 每步 FSDP checkpoint ≈36G（optim 23G + model 13G），80G 数据盘仅容 2 个完整 checkpoint，verl 恒保存 optimizer 且 `total_training_steps=1` 仍落 2 个 checkpoint；smoke 后删 optimizer 状态、保留 4 个 model 权重（GRPO/Tree 各 2 step）供 T09 导出。结果表/失败案例/资源估算已回填 `docs/researchsearch/week1-report.md` §5–§7。
- [x] **T09 checkpoint 导出验证 + `model_merger.py` 兼容修复**：发现并修复 `scripts/merge_ckpt/model_merger.py` 与 verl 0.8.5 的两个兼容性 bug——① `from verl.utils import hf_processor` 在 verl 0.8.5 无此符号（改 `try/except` 兜底 `None`，纯文本模型不需要 processor）；② verl 0.8.5 把 config/tokenizer 存到 `{local_dir}/huggingface/` 子目录，merger 原先 `AutoConfig.from_pretrained(local_dir)` 读不到（新增 `_resolve_hf_config_path` 优先指向 huggingface/ 子目录）。修复后用 `export_ckpt.sh --merge` 将 tree step1 完整 checkpoint 导出为 HF（`model.safetensors` 2 分片共 6.8G + config/tokenizer/generation_config），`AutoModelForCausalLM.from_pretrained` 重载成功（qwen2 / 3.09B 参数），生成 `"The largest planet ... is Jupiter ..."` 正确。tree step2 因磁盘满缺 `huggingface/` 子目录不可导出，仅 3 个完整 checkpoint（grpo step1/2、tree step1）。随后清理 39G（删 tree step2 残骸、已导出的 tree step1、重复的 grpo step1），并导出 grpo step2 → `merged/grpo-step2-hf`（重载 3.09B 参数成功）。现数据盘 32G used / 49G free，保留 grpo step2 原始 FSDP 分片 + 两个导出 HF（tree-step1-hf、grpo-step2-hf）+ 原始模型 models/。

### 测试状态

234 项（233 通过 + 1 xfail）：`data_build` 14（T01 + FlashRAG split 锁定）+ `reward_format` 14（T04）+ `evidence` 22（T05 + Recall@K + 截断后可见判定）+ `core_algos` 9（T07）+ `retrieval_client` 16（T03 客户端）+ `retrieval_server` 11（T03 服务端格式层，1 xfail 记录 return_scores=false 缺陷）+ `loss_mask` 10（T06 chain + tree）+ `adv_routing` 4（T07 路由）+ `tree_evidence_wiring` 12（Day 6 接线）+ `export_ckpt` 5（T09 脚本）+ `e0_export` 8 + `tree_export` 8 + `tree_shared_prefix` 4 + `web_cache_budget` 19（B-5 命名空间 + B-6 预算口径 + 缺口 2/3 失败分类与旁路）+ `cost_stats` 13（T10 统一口径：六口径字段 / 累加器 / 共享前缀物理 token / checkpoint 体积 / 检索侧逻辑搜索与外部请求分开）+ `eval_metrics` 27（E06 评测评分器 + E07 消融：归一化一致 / EM/F1 别名 / 宏平均 / Multi-Hop Avg EM / bootstrap CI / pairwise_ablation）+ `eval_runner` 16（Day 7 评测编排 + E06/E07：schema 校验 / EM 独立于训练奖励 / Recall@K 与证据集合分离 / 可映射比例分母 / Multi-Hop Avg EM 缺项不静默 / 消融题目配对）+ `eval_manifest` 11（Day 7 manifest 预注册骨架：四数据集完整性强制 / 只读哈希 / 篡改校验 / 摘要）+ `evidence_map` 11（Day 1 evidence_map_fn：四态判定 / 歧义不猜测 / 语料缺失不误判 / 去重）。

### 门禁对照

| 门禁 | 状态 | 说明 |
|---|---|---|
| T01 分组/索引 | ✅ | build.py 全局唯一 index + dev split 锁定（不消费 test），data_build 覆盖 |
| T02 证据映射 | ⛔ | 依赖真实 FlashRAG 语料/索引 |
| T03 检索契约 | 🟡 | 客户端超时/重试/错误分类（16 项）+ 服务端格式层契约用 fake retriever 交付（11 项，含 1 xfail 缺陷记录）；批量结果顺序稳定、doc_id 回查正文仍需真实索引 |
| T04 分段奖励 | ✅ | 14 项测试锁定语义 |
| T05 证据奖励 | ✅ | evidence.py + 22 项测试（R_evidence 8 + Recall@K 5 + 截断后可见判定 9）+ doc_id 提取纯函数 + 树内接线测试（tree_evidence_wiring 12） |
| T06 loss mask | ✅ | 静态分析 + 单元测试完成（10 项：chain 6 + tree 4）+ Tree smoke 运行时 `state_tokens/coverage 0.771` 观察通过 |
| T07 优势计算 | ✅ | 9 项 core_algos + 4 项路由测试锁定 |
| T08 权重更新 | ✅ | GRPO 2 步 / Tree 1 步真实 optimizer 更新，`grad_norm` 非零，checkpoint 保存（见 week1-report §5） |
| T09 checkpoint | 🟡 | 导出脚本参数化（5 项）+ `model_merger.py` 兼容 verl 0.8.5 修复；真实导出/加载已验证（tree step1→HF，重载 3.09B 参数、推理 Jupiter 正确）；Web warm start（下一阶段更新）待真实检索 |
| T10 预算 | 🟡 | 缓存命名空间 + 三口径统计（logical_queries/cache_hits/api_requests）+ 上限终止已用 fixture 测试锁定（web_cache_budget 19）；统一六口径 `cost_stats`（CostStats / 共享前缀物理 token 纯函数 / checkpoint 体积 / 检索侧逻辑搜索与外部请求分开，13 项，见 [cost-stats.md](./cost-stats.md)）；真实 rollout 填数与 GPU 时间仍待 GPU+检索服务 |

### 剩余阻塞项

- [x] 核验 Ray runtime_env 在 worker 上分发代码及模型、数据路径（三段 smoke 均经 Ray job submit + runtime_env 跑通，代码/模型/数据路径分发验证通过）。
- [ ] 补装 retriever 环境 faiss-gpu（1.7.3 为版本笔误，改用 PyPI 可用版本 1.7.2 / 1.14.3 等），随后配置检索/Web 服务并记录一次真实搜索请求。
- [ ] 真实 FlashRAG 语料/索引就绪后：生成 128/64 条 smoke parquet + manifest，补齐 `doc_id` 与证据映射报告。
- [ ] 检索服务就绪后：执行真实检索层契约测试（批量顺序稳定、doc_id 回查正文、中英文字符、服务不可用）；格式层契约已用 fake retriever 交付（`test_retrieval_server.py`）。
- [ ] 证据奖励真实 rollout 验证（Day 6 后半）：无卡接线已落地（pop 增字段 → 逐叶 `visible_doc_ids` → RewardManager 证据项，见 `docs/researchsearch/evidence-wiring.md`）；真实 rollout 逐叶验证、`alpha>0` 更新、「截断后可见性判定」待 GPU+语料+检索服务。
- [x] loss mask 运行时断言（真实 rollout）：Tree smoke 运行时 `state_tokens/coverage 0.771` 观察通过（tree 路径；chain 路径由 10 项单测覆盖）。
