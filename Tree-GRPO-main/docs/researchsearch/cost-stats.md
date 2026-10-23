# 成本统计口径（T10）

版本：v1.0 · 2026-10-22
对应代码：`search_r1/search/cost_stats.py`、`search_r1/search/retrieval_client.py::post_retrieve`
对应方案：《第 2 周 Wiki 实验开发方案》§6.3、门禁 T10；架构 §3「成本口径」。

本文定义六口径的**单一权威字段名与语义**，作为 Wiki / Web / 训练三侧共用口径。
无卡无语料阶段交付「统计代码 + 口径」；真实训练时按 §6 填数清单接入并上报。

## 1. 六口径权威定义

| 口径 | 字段名 | 定义与粒度 |
|---|---|---|
| 物理新生成 token | `physical_new_tokens` | 模型在 rollout 中新生成的 token 数。**共享前缀（父节点那一段）只计一次**；工具返回 `<information>` 与 padding 不计入（architecture：工具返回内容不作为模型生成 token）。 |
| 逻辑搜索次数 | `logical_searches` | Agent 调用检索工具的次数（**动作粒度**）。一次 `_batch_search` / `post_retrieve` = 1 次，可含多个 query。 |
| 实际外部请求 | `http_requests` | 底层 HTTP 请求总数，**含重试**；每次网络尝试（成功 / 失败 / 超时）都计一次。 |
| 缓存命中 | `cache_hits` | 命中缓存、未发出外部请求的查询数。 |
| GPU 时间 | `gpu_time_seconds` | 训练 / 推理占用 GPU 的墙钟时间（秒）；GPU 小时 = `/ 3600`。 |
| checkpoint 体积 | `checkpoint_bytes` | 每次保存 checkpoint 的磁盘字节数（FSDP 分片合计）。 |

「逻辑搜索」与「实际外部请求」**必须分开报告**（架构 §3「区分逻辑调用与真实请求」）。
轨迹条数相同不代表成本相同——Tree 的 `ts_k` 是每树抽样叶子数，不等于独立轨迹数或物理生成成本。

## 2. 与已落地代码的映射

Web 侧 `bing_search_server_redis.py::get_stats()` 三口径**不重构**，聚合时按此映射换算：

| 统一口径字段 | Web 侧字段 | 换算 |
|---|---|---|
| `logical_searches`（动作粒度） | `logical_queries`（query 粒度） | 1 次 `/retrieve` 端点请求 = 1 次逻辑搜索，可含多个 query |
| `http_requests` | `api_requests` | 同义（均含重试） |
| `cache_hits` | `cache_hits` | 同义 |

Wiki 侧（本地 FAISS/E5 检索）无缓存，`cache_hits` 恒为 0；真实外部请求计数由客户端侧 `post_retrieve` 提供（§3）。

## 3. 代码位置

- `search_r1/search/cost_stats.py`
  - `COST_FIELDS`：六口径规范字段名（单一权威定义）。
  - `CostStats`：进程内线程安全累加器（`logical_search` / `http_request` / `cache_hit` /
    `add_physical_new_tokens` / `add_gpu_seconds` / `add_checkpoint_bytes` / `merge` / `reset` / `as_dict`）。
  - `physical_new_tokens_from_tree(nodes, pad_token_id)`：纯函数，累加树上每个节点的增量 token。
  - `directory_size_bytes(path)`：递归统计目录体积。
- `search_r1/search/retrieval_client.py::post_retrieve(..., stats=None)`：可选累加器参数。
  一次调用记 1 次逻辑搜索；每次网络尝试记 1 次实际请求（含重试、连接失败与超时）。
  默认 `None` 向后兼容，不统计。

## 4. 物理新生成 token 口径（共享前缀只计一次）

树节点存的是**增量** `responses_with_info_mask`——本节点从父状态新生成的那一段，
`<information>` 位置已用 `pad_token_id` 掩掉。因此父节点那一段只在其自身节点计一次，
子节点不复生成，天然满足「共享前缀不计入物理新生成 token」。

`physical_new_tokens_from_tree` 遍历 `root.get_subtree_nodes()`（根节点的 responses
通常为空），对每个节点的 `responses_with_info_mask` 数 `!= pad_token_id` 的 token 数求和。
该函数只访问节点属性、不 import 树实现，便于离线单测。

## 5. 检索侧统计口径

- **逻辑搜索**：`post_retrieve` 进入时 `stats.logical_search()`（每次调用 +1，无论批量含几个 query）。
- **实际外部请求**：for 重试循环内、`requests.post` 前 `stats.http_request()`（每次尝试 +1，
  连接失败 / 超时同样计入，因为它也是一次发出的外部请求）。
- 一次逻辑搜索因重试可产生 1..(max_retries+1) 次实际请求，二者分开计数。

## 6. 真实训练填数清单（接线已落地，真实数值待 GPU）

接线已按下列埋点落地（rollout 侧 `generation_ts` + trainer 侧 `ray_trainer_ts` + 检索侧
`post_retrieve`）；`CostStats` 随训练 metrics 上报，真实数值待 GPU / 语料训练回填。口径定义不变：

| 口径 | 埋点位置 | 接入方式 |
|---|---|---|
| `physical_new_tokens` | Tree 树建完后 / rollout 循环 | `physical_new_tokens_from_tree(root.get_subtree_nodes(), tokenizer.pad_token_id)`，累加进 worker 的 `CostStats` |
| `logical_searches` / `http_requests` | rollout 侧 `_batch_search` | 持有 `CostStats` 实例并传给 `post_retrieve(stats=...)` |
| `cache_hits` | Wiki 恒 0；Web 取 `get_stats()` | 服务端聚合后换算（§2） |
| `gpu_time_seconds` | trainer `fit` 循环 | 稳态秒/step × step × GPU 数（方案 §11），或真实墙钟计时 |
| `checkpoint_bytes` | `save_checkpoint` 后 | `directory_size_bytes(ckpt_dir)` 写入日志 |

第 1 周单卡 FSDP checkpoint ≈36G/step 的磁盘压力在 8-GPU 分片下不复现，但仍须在
日志记录 checkpoint 体积与磁盘占用（方案 §6.3）。

## 7. 聚合与上报

每个 Ray rollout worker 持有一个 `CostStats`；driver 用 `merge` 聚合所有 worker，
再并入 Web 服务端 `get_stats()` 按 §2 换算后的值，写入运行日志 / 周报。训练与推理
分开报告（架构 §3「训练与推理分开」）。

## 8. 测试与验收

- `tests/researchsearch/test_cost_stats.py`（13 项）锁定：六口径字段完整、累加器
  `merge` / `reset` / 线程安全、共享前缀只计一次、padding / information 不计入、
  checkpoint 体积、`post_retrieve` 逻辑搜索与外部请求（含重试）分开。
- 门禁 T10「成本统计齐全（物理生成 token、逻辑搜索、外部请求、GPU 时间）」在真实
  训练后按 §6 填数转 ✅；接线已落地（§6 各埋点 + 训练 metrics 上报），无卡阶段
  交付的是统计代码、口径与接线，不伪称真实数值已产出。
