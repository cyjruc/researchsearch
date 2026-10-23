# Wiki / Web 检索契约

对齐《第 1 周工程闭环开发方案》§4.3。本文记录当前源码已实现的协议，并标注与方案目标的缺口；标注"缺口"的项尚未接通，不能视为已满足。

## 1. 统一调用协议

Agent rollout 发出的请求形状（`generation.py` / `generation_ts.py` 的 `_batch_search`）：

```json
{"queries": ["查询1", "查询2"], "topk": 3, "return_scores": true}
```

| 字段 | 语义 |
|---|---|
| `queries` | 批量查询；返回外层顺序与之一一对应 |
| `topk` | 每个查询返回的文档数 |
| `return_scores` | 是否返回相似度分数（rollout 恒为 `true`） |

批量请求与返回的外层顺序必须一一对应；重复查询各自独立返回，不得去重合并。

## 2. Wiki 后端（FAISS/E5 或 BM25）

服务：`search_r1/search/retrieval_server.py` 的 `POST /retrieve`；检索器在 `search_r1/search/retrieval.py`。

`return_scores=true` 时返回形状：

```json
{"result": [
  [
    {"document": {"title": "标题", "text": "正文", "contents": "标题\n正文"}, "score": 0.81},
    {"document": {"title": "...", "text": "...", "contents": "..."}, "score": 0.79}
  ]
]}
```

- 外层 `result` 长度 == `queries` 长度。
- 内层每个元素是 `{"document": {...}, "score": <float>}`。
- `document.contents` 的第一行是标题，其余是正文；`_passages2string` 据此把模型可见内容格式化为 `Doc {i}(Title: {title}) {text}`。
- 无结果返回 `[]`（不是 `null`）；不足 `topk` 条时返回实际条数。

### 已知缺口（方案 §4.3，需在语料层补齐）

1. **`document` 无稳定 `id`，doc_id 采用 title 级**（第 2 周 §3.2 决策）：证据奖励的"文档"粒度为 title 级（架构 §2.1：文档级证据覆盖，不代表句子级证据充分性），`doc_id = wiki:<title>`。wiki-18 是段落级切分，同一 title 对应多个段落行（chunk），须按 title 归并为同一文档。真实返回结构：BM25 返回 `{"title", "text", "contents"}`（无 id）；Dense/FAISS 返回 `{"id": "<数字行号>", "contents"}`（id 是段落级 chunk 行号，非文档级）。**已修复**：`extract_doc_ids_from_retrieval` 只把 `wiki:` 前缀的字符串 `document.id` 当稳定 doc_id，其余（整数行号、字符串数字行号如 `"0"`、缺失）一律从 `document.title`（BM25）或 `document.contents` 首行去引号（Dense）构造 `wiki:<title>`；由 `tests/researchsearch/test_evidence.py::test_extract_doc_ids_dense_string_row_id_falls_back_to_title` 等锁定。**离线契约验证**（2026-10-23）：用 wiki-18 前 2000 行真实语料行喂 `extract_doc_ids_from_retrieval`，2000/2000 行 doc_id 正确为 `wiki:<去引号title>` 且 100% 命中 `title_docid_map` 的 gold doc_id。**端到端真实检索仍阻塞**：E5/FAISS Dense 检索器 `model.cuda()` + `faiss_gpu` 强依赖 GPU，当前 AutoDL 实例无 GPU（`No devices were found`），检索服务起不来；待 GPU 开机后按 §7 做真实检索验收（批量顺序稳定、doc_id 可回查正文、重复查询可复核）。
2. **语料与索引条目映射**：需核验向量维度、语料条数、索引条目与 `contents` 的一一对应（Day 2 任务 1）。

## 3. Web 后端（Bing / Brightdata + Redis）

服务：`search_r1/search/bing_search_server_redis.py` 的 `POST /retrieve`。

返回形状（字符串列表）：

```json
{"result": ["Page 1: 摘要1", "Page 2: 摘要2"]}
```

- 外层 `result` 长度 == `queries` 长度，每个元素是格式化字符串，行格式 `Page {i}: {snippet}`。
- `max_results` / `result_length` 是**服务启动参数**，请求中的 `topk` 不参与当前 endpoint 处理（`SearchRequest` 只接受 `queries`）。
- 缓存键已命名空间化：`{provider}:{zone}:{cache_version}:{max_results}:{result_length}:{query}`（`OnlineSearchEngine._cache_key`），换 provider/zone/版本/topk 后旧缓存不再命中。

### 已知缺口（方案 §4.3）

1. ~~缓存键不含 provider/topk/版本~~（已修复）：缓存键现由 `provider` / `zone` / `cache_version` / `max_results` / `result_length` / `query` 组成，换任一配置后不再命中旧缓存；由 `tests/researchsearch/test_web_cache_budget.py` 锁定。注意：键格式变更使历史 JSON 持久化缓存（裸 query 键）不再命中，属预期行为。
2. ~~失败以普通文本返回~~（已修复，服务端引擎层）：`OnlineSearchEngine` 现通过 `get_last_outcome()` 暴露结构化 `status`（`success`/`not_found`/`rate_limit`/`network`/`server`/`invalid`/`auth`/`empty_query`）与 `error_code`，能区分限流、网络失败与真实零结果；鉴权错误（401/403）立即失败不重试。`execute`/`execute_with_retry` 的返回字符串保持兼容不变。注意：FastAPI `/retrieve` 端点仍返回兼容字符串列表，结构化状态尚未串入响应或服务端日志（待真实服务接入时决定暴露形式）。
3. ~~不保留 URL / 标题 / 原始排名~~（已修复，服务端引擎层）：`SearchOutcome.items` 保留每条 organic 的 `rank`/`title`/`url`/`description` 旁路元数据（URL/标题仅在服务提供时保留，缺失为空字符串），模型可见文本仍只含摘要，URL 不进摘要；实际请求计数由 `get_stats()` 三口径提供。

## 4. 模型可见内容 vs 旁路元数据

| | 模型观察 | 旁路（证据/日志） |
|---|---|---|
| Wiki | `Doc {i}(Title: ...) {text}` | `doc_id`、`score`、检索轮次与原始排名 |
| Web | `Page {i}: {snippet}` | 原始 rank、错误状态、请求计数、缓存命中、URL（若提供） |

证据奖励基于旁路保留的 `doc_id` 计算，模型可见内容只是标题与正文/摘要，二者分开留档。

## 5. 超时与重试

- **方案目标**：单次请求 30 秒、最多重试 2 次、指数退避；鉴权错误立即失败；客户端与服务端共同约束总重试预算，避免嵌套重试放大请求数。
- **已修复（原方案 §2.1 阻塞 6）**：两个 rollout 的 `_batch_search` 都已改走 `search_r1/search/retrieval_client.py::post_retrieve`，统一 30s 超时、`max_retries` 重试与错误分类（`network`/`auth`/`rate_limit`/`server`/`invalid`），失败抛 `RetrievalError`；由 `tests/researchsearch/test_retrieval_client.py`（11 项）锁定。服务端 `tool_retry_count` 仍存在，客户端与服务端的**总重试预算尚未统一**（§4.3 剩余项，避免嵌套重试放大请求数）。

## 6. 错误分类

方案要求区分：真实零结果、限流、网络失败、非法返回、鉴权失败。健康预检失败时禁止启动正式训练；运行中系统错误导致的轨迹标记无效，不静默当作答案错误计入训练。

## 7. 测试执行状态（格式层已交付；检索层仍需真实服务）

**已交付（本地无卡，fake retriever）**：`tests/researchsearch/test_retrieval_server.py`（10 项 + 1 xfail）锁定 endpoint 格式层契约——返回形状、外层顺序对应、重复查询不合并、空结果 `[]`、不足 `topk` 返回实际条数、`topk` 缺省回退、`document` 结构（含 `id`）透传、非法输入（缺 `queries` / 非 list / 元素非 str）被 Pydantic 拒绝。真实索引就绪后，把 fake retriever 换成真实检索器即可复用。

**仍需真实服务（语料/索引就绪后）**：单条与批量真实检索、中英文字符编码、服务不可用（连接拒绝 / 超时）。验收：批量结果顺序稳定、`doc_id` 可回查正文、同一冻结环境重复查询可复核。

**已知缺陷（不伪报）**：`retrieve_endpoint` 的 `results, scores = batch_search(...)` 解包与 retriever 在 `return_score=False` 只返回单值不兼容，`return_scores=false` 请求会解包失败或错位；rollout 恒 `return_scores=true` 不触发，由 `test_return_scores_false_should_return_docs_without_scores`（xfail）锁定。修复方向：让 `batch_search` 始终返回 `(results, scores)`（`return_score=False` 时 scores 为空列表），或端点按 `return_scores` 分支处理。

## 8. 源码位置

- 请求/返回处理：`search_r1/llm_agent/generation.py`、`generation_ts.py`（`_batch_search` / `_passages2string`）
- Wiki 服务：`search_r1/search/retrieval_server.py`、`search_r1/search/retrieval.py`
- Web 服务：`search_r1/search/bing_search_server_redis.py`
