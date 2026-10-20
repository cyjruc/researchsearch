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

1. **`document` 无稳定 `id`**：当前 BM25/E5 检索器只返回 `title` / `text` / `contents`，没有可回查正文的稳定文档 ID。证据奖励所需 `doc_id`（形如 `wiki:<docid>`）必须从语料/索引条目映射中核验并补齐，不能依赖字符串里的 "Doc 1/2"（这些编号在不同搜索中重复）。
2. **语料与索引条目映射**：需核验向量维度、语料条数、索引条目与 `contents` 的一一对应（Day 2 任务 1）。

## 3. Web 后端（Bing / Brightdata + Redis）

服务：`search_r1/search/bing_search_server_redis.py` 的 `POST /retrieve`。

返回形状（字符串列表）：

```json
{"result": ["Page 1: 摘要1", "Page 2: 摘要2"]}
```

- 外层 `result` 长度 == `queries` 长度，每个元素是格式化字符串，行格式 `Page {i}: {snippet}`。
- `max_results` / `result_length` 是**服务启动参数**，请求中的 `topk` 不参与当前 endpoint 处理（`SearchRequest` 只接受 `queries`）。
- 缓存键当前仅为 `query`（Redis `get/set` 直接使用 query 字符串）。

### 已知缺口（方案 §4.3）

1. **缓存键不含 provider/topk/版本**：换 provider、改 `max_results` 或版本后，旧缓存仍会命中。需把 provider、topk、版本纳入缓存键。
2. **失败以普通文本返回**：`execute_with_retry` 最终返回 `"Search failed for query: {query}"`，没有结构化 `status` / `error_code`，无法区分限流、网络失败与真实零结果。
3. **不保留 URL / 标题 / 原始排名**：仅从 organic 条目提取 `description`。方案要求保留原始 rank、错误状态与实际请求计数作为旁路日志；若服务提供 URL，仅作旁路元数据，不改变模型可见摘要。

## 4. 模型可见内容 vs 旁路元数据

| | 模型观察 | 旁路（证据/日志） |
|---|---|---|
| Wiki | `Doc {i}(Title: ...) {text}` | `doc_id`、`score`、检索轮次与原始排名 |
| Web | `Page {i}: {snippet}` | 原始 rank、错误状态、请求计数、缓存命中、URL（若提供） |

证据奖励基于旁路保留的 `doc_id` 计算，模型可见内容只是标题与正文/摘要，二者分开留档。

## 5. 超时与重试

- **方案目标**：单次请求 30 秒、最多重试 2 次、指数退避；鉴权错误立即失败；客户端与服务端共同约束总重试预算，避免嵌套重试放大请求数。
- **当前缺口**（方案 §2.1 阻塞 6）：rollout 的 `_batch_search` 直接 `requests.post(...).json()`，**无客户端超时、无 HTTP 状态检查**。Web 服务端有 `tool_retry_count` 重试，但客户端无保护，服务端超时不能替代客户端保护。

## 6. 错误分类

方案要求区分：真实零结果、限流、网络失败、非法返回、鉴权失败。健康预检失败时禁止启动正式训练；运行中系统错误导致的轨迹标记无效，不静默当作答案错误计入训练。

## 7. 待执行测试清单（需真实服务，属 Day 2 卡住部分）

服务与索引就绪后执行，覆盖：单条查询、批量查询、中英文字符、空结果、非法输入（缺字段 / 非 list）、服务不可用（连接拒绝 / 超时）。验收：批量结果顺序稳定、`doc_id` 可回查正文、同一冻结环境重复查询可复核。

## 8. 源码位置

- 请求/返回处理：`search_r1/llm_agent/generation.py`、`generation_ts.py`（`_batch_search` / `_passages2string`）
- Wiki 服务：`search_r1/search/retrieval_server.py`、`search_r1/search/retrieval.py`
- Web 服务：`search_r1/search/bing_search_server_redis.py`
