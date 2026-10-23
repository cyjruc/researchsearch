# 树内证据奖励接线记录（阻塞 8 / 已知陷阱 #2）

更新时间：2026-10-22

本文记录证据奖励 `R_evidence` 接入树内 `reward_fn(final_output)` 的字段链路、三个
丢失点与落地实现。**无卡接线已完成并经 fixture 测试锁定**；真实 rollout 的逐叶验证
仍待补。「截断后可见性判定」纯函数已交付并经单元测试锁定，真实 rollout 接线待接入
（见 §6），未做项不标记为已接通。

参照：《项目架构与实验设计》§2.1、§4.4；《第 1 周工程闭环开发方案》§4.2/§4.4/§6 Day 6。

## 1. 字段传递全链路

数据从 parquet 到树内奖励回调，经过四层：

```text
rl_dataset.__getitem__          ray_trainer_ts.fit            generation_ts.run_llm_loop_tree_search        RewardManager.__call__
row_dict(含 extra_info)  ──▶    batch.pop(...)  ──▶  gen_batch.repeat(k)  ──▶  final_output  ──▶  score
```

逐层源码：

1. [rl_dataset.py:152-153](../verl/utils/dataset/rl_dataset.py#L152-L153)：`extra_info`
   作为 parquet 的 dict 列原样保留在 `row_dict`，并单独提取 `index`。`collate_fn`
   后 `extra_info` / `index` / `reward_model` / `data_source` 都进入
   `DataProto.non_tensor_batch`。
2. [ray_trainer_ts.py:842-849](../verl/trainer/ppo/ray_trainer_ts.py#L842-L849)：
   `batch = DataProto.from_single_dict(batch_dict)` → `repeat(ts_m)` →
   `uid = index.copy()` → `gen_batch = batch.pop(...)`。**已改为在 `extra_info` 存在时
   将其一并加入 `non_tensor_batch_keys`**（条件性，避免 `DataProto.pop` 对缺失 key 断言）。
3. [generation_ts.py:570](../search_r1/llm_agent/generation_ts.py#L570)：
   `final_output = gen_batch.repeat(k, interleave=True)`，`non_tensor_batch` 用
   [verl/protocol.py:586-591](../verl/protocol.py#L586-L591) 的 `np.repeat(val, k)`
   复制，因此 `extra_info` 随根样本逐份重复 `k` 次，行序与叶子一一对齐。
4. [generation_ts.py:683](../search_r1/llm_agent/generation_ts.py#L683)：
   `original_scores, _ = self.reward_fn(final_output)` 在树采样**之后、与原 batch 合并之前**
   调用，可读到 `gen_batch` 带进来的 `extra_info` 与 rollout 写入的 `visible_doc_ids`。

## 2. 三个丢失点（均已修）

| # | 位置 | 原丢失内容 | 落地改法 |
|---|---|---|---|
| 1 | [ray_trainer_ts.py:843-849](../verl/trainer/ppo/ray_trainer_ts.py#L843-L849) | `batch.pop` 不传 `extra_info`，证据字段进不了 `gen_batch` | 条件性把 `extra_info` 加入 `non_tensor_batch_keys` |
| 2 | [generation_ts.py](../search_r1/llm_agent/generation_ts.py) `batch_search` / `execute_predictions` | `_passages2string` 只用 `contents`，`document.id` 被丢弃 | `batch_search` 返回 `(formatted, doc_ids)`；`TreeNode` 新增 `visible_doc_ids` 沿父链累积；`run_llm_loop_tree_search` 在 `reward_fn` 之前写入 `final_output.non_tensor_batch['visible_doc_ids']` |
| 3 | [main_ppo_format_ts.py](../verl/trainer/main_ppo_format_ts.py) `RewardManager` | 只算答案+格式分，无证据项 | 新增 `evidence_alpha`（默认 0），逐条读 `extra_info.supporting_doc_ids` / `evidence_status` + `visible_doc_ids`，调 `compute_evidence_reward` + `combine_score_with_evidence` |

## 3. 关键不变量：叶子顺序与 final_output 行序对齐

接线正确性的前提是「叶子顺序 == final_output 行序」。该对齐由两层机制共同保证：

- `final_node_list` 按 root 分组：对每个 root `sample_leaf(k)` 后 `extend`，得到
  `[root0 的 k 叶, root1 的 k 叶, ...]`；`sample_leaf` 返回 `get_subtree_nodes` 的
  BFS 顺序叶子（与 `random.shuffle` 无关）。
- `final_output = gen_batch.repeat(k, interleave=True)` 用 `np.repeat`，行序为
  `[根样本0 ×k, 根样本1 ×k, ...]`。

因此证据字段随 `gen_batch.repeat(k)` 传播后，其第 `i` 行与 `final_node_list[i]`
所属的根样本一一对应（`i // k` 定位根样本）。`final_output.batch['responses']` 直接
由 `final_node_list` 的 `node.responses` 拼成，对应关系由构造保证。**已由
`tests/researchsearch/test_tree_evidence_wiring.py` 的
`test_leaf_order_aligns_with_interleaved_repeat` 锁定。**

## 4. 落地实现要点

### 4.1 `extra_info` 随 gen_batch 传入（丢失点 1）

[ray_trainer_ts.py:842-849](../verl/trainer/ppo/ray_trainer_ts.py#L842-L849)：

```python
non_tensor_batch_keys = ['uid', 'data_source', 'reward_model']
if 'extra_info' in batch.non_tensor_batch:
    non_tensor_batch_keys.append('extra_info')
gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                      non_tensor_batch_keys=non_tensor_batch_keys)
```

整列 `extra_info` dict 随树展开重复（每叶一份），在奖励回调内按 key 取窄字段；
比「提取独立窄字段」改动更小，代价是 dict 列随 `k` 展开的内存开销（本周 smoke 规模可接受）。

### 4.2 旁路保留文档 ID 并逐叶收集（丢失点 2）

- `batch_search` 返回 `(formatted, doc_ids)`，`doc_ids` 来自
  `evidence.extract_doc_ids_from_retrieval`（保序、跳过缺失 id）。
- `execute_predictions` 返回 5 元组，每样本本轮的 `search_doc_ids` 随检索分支累加。
- `TreeNode` 新增 `visible_doc_ids`（默认独立空列表），子节点构造时
  `parent.visible_doc_ids + search_doc_ids[i]`，共享前缀检索结果自然复用。
- `run_llm_loop_tree_search` 在收集 `final_node_list` 后、**`reward_fn` 之前**，把
  `final_visible_doc_ids_list` 写入 `final_output.non_tensor_batch['visible_doc_ids']`
  （`np.empty(n, dtype=object)` + 切片赋值，避免等长列表被压成二维 object 数组）。

### 4.3 奖励回调计算证据项（丢失点 3）

[main_ppo_format_ts.py](../verl/trainer/main_ppo_format_ts.py) 的 `RewardManager`：

```python
extra_info = data_item.non_tensor_batch.get('extra_info') or {}
gold_doc_ids = extra_info.get('supporting_doc_ids', [])
evidence_status = extra_info.get('evidence_status', 'mapped')
visible_doc_ids = data_item.non_tensor_batch.get('visible_doc_ids', [])
evidence_res = compute_evidence_reward(gold_doc_ids=gold_doc_ids,
                                       visible_doc_ids=visible_doc_ids,
                                       evidence_status=evidence_status)
score = combine_score_with_evidence(score, evidence_res, self.evidence_alpha)
```

`evidence_alpha` 来自 `config.reward_model.evidence_alpha`（[ppo_trainer.yaml](../verl/trainer/config/ppo_trainer.yaml)
新增，默认 `0.0` 关闭）。字段缺失时安全降级为证据项 0，不影响原奖励。

## 5. 已交付组件与测试

| 组件 | 位置 | 覆盖 |
|---|---|---|
| doc_id 提取 / 证据奖励 / 加权合并 | `verl/utils/reward_score/evidence.py` | `extract_doc_ids_from_retrieval`、`compute_evidence_reward`、`combine_score_with_evidence`（纯 stdlib，无 GPU 可测） |
| 接线不变量测试 | `tests/researchsearch/test_tree_evidence_wiring.py`（12 项） | 叶子顺序对齐、doc_id 提取、两题不串题、`combine_score_with_evidence` 的 alpha 语义、`visible_doc_ids` 默认隔离与导出拷贝、端到端证据合并 |
| 优势路由测试 | `tests/researchsearch/test_adv_routing.py`（4 项） | `compute_advantage` 的 `tree`/`tree_2norm` 路由 |

## 6. 尚待验证 / 未实现（未伪报）

1. **真实 rollout 验证**（需 GPU + 语料 + 检索服务）：`extra_info` 与
   `visible_doc_ids` 在真实树采样中确实随 `gen_batch.repeat` 到达树内 `reward_fn`，
   且两题证据不同时预期分数改变、不串题（方案 §6 Day 6 任务 1 验收）。
2. **截断后可见性判定（判定函数已交付，真实 rollout 接线待接入）**：当前
   `visible_doc_ids` 仍 = 后端返回的全部 doc_id（「返回即视为可见」）。方案 §4.2/§4.4
   要求「截断后仍进入 Agent 观察才算可见」，被 `max_obs_length` 截掉的文档应记入
   `retrieval_hit_doc_ids`、不计入分子。判定纯函数
   `evidence.split_doc_ids_by_truncation`（`visible` 与 `retrieval_hit` 互斥划分，
   结束位置 `<= max_obs_length` 即完整保留）已交付，并由
   `tests/researchsearch/test_evidence.py` 的 9 项测试锁定（含前缀 offset、部分截断、
   长度不一致抛错、端到端喂给 `compute_evidence_reward`）。**尚未接入 rollout 热路径**：
   `execute_predictions` 的 `search_doc_ids` 仍需改为「按每个文档在格式化 observation
   中的 token 边界 + `max_obs_length` 调 `split_doc_ids_by_truncation` 过滤」，这一步
   需要 tokenizer 对每个 `_passages2string` 片段计数（`doc_token_lengths`）与
   `\n\n<information>` 前缀 token 数，属真实语料/检索就绪后的接入项。
3. **Wiki 稳定 id 前置缺口**：`extract_doc_ids_from_retrieval` 依赖
   `document.id`，而当前 Wiki 检索器只返回 `title`/`text`/`contents`，见
   [retrieval-contract.md](./retrieval-contract.md) §2。语料层补齐稳定 id 前，
   真实 Wiki 证据奖励读到的 `doc_id` 恒为空。
