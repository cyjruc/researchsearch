# 树内证据奖励接线蓝图（阻塞 8 / 已知陷阱 #2）

更新时间：2026-10-20

本文记录证据奖励 `R_evidence` 接入树内 `reward_fn(final_output)` 的完整字段链路、
三个丢失点与 Day 6 接线方案。**本文为静态分析与蓝图，核心训练路径改动尚未落地，
须在真实 rollout 上验证后才可标记「证据奖励已接通」。** 已交付的预备组件（纯函数
`evidence.extract_doc_ids_from_retrieval` 与两个固定测试样例文件）不依赖 GPU。

参照：《项目架构与实验设计》§2.1、§4.4；《第 1 周工程闭环开发方案》§4.2/§4.4/§6 Day 6。

## 1. 现状：字段传递全链路

数据从 parquet 到树内奖励回调，实际经过四层：

```text
rl_dataset.__getitem__          ray_trainer_ts.fit            generation_ts.run_llm_loop_tree_search        RewardManager.__call__
row_dict(含 extra_info)  ──▶    batch.pop(...)  ──▶  gen_batch.repeat(k)  ──▶  final_output  ──▶  score
```

逐层源码：

1. [rl_dataset.py:152-153](../verl/utils/dataset/rl_dataset.py#L152-L153)：`extra_info`
   作为 parquet 的 dict 列原样保留在 `row_dict`，并单独提取 `index`。`collate_fn`
   后 `extra_info` / `index` / `reward_model` / `data_source` 都进入
   `DataProto.non_tensor_batch`。
2. [ray_trainer_ts.py:837-846](../verl/trainer/ppo/ray_trainer_ts.py#L837-L846)：
   `batch = DataProto.from_single_dict(batch_dict)` → `repeat(ts_m)` →
   `uid = index.copy()` → `gen_batch = batch.pop(batch_keys=['input_ids','attention_mask','position_ids'],
   non_tensor_batch_keys=['uid','data_source','reward_model'])`。
   **`extra_info` 没有被 pop，进不了 `gen_batch`。**
3. [generation_ts.py:562-563](../search_r1/llm_agent/generation_ts.py#L562-L563)：
   `final_output = gen_batch.repeat(k, interleave=True)`，`non_tensor_batch` 用
   [verl/protocol.py:586-591](../verl/protocol.py#L586-L591) 的 `np.repeat(val, k)`
   复制，因此 `uid` / `data_source` / `reward_model` 会按根样本逐份重复 `k` 次。
4. [generation_ts.py:668](../search_r1/llm_agent/generation_ts.py#L668)：
   `original_scores, _ = self.reward_fn(final_output)` 在树采样**之后、与原 batch 合并之前**
   调用，能读到的只有 `gen_batch` 带进来的字段。

结论：证据字段在四个入口（`sample_id` / `supporting_doc_ids` / `evidence_status` /
`corpus_version`）里目前一个都到不了 `final_output`。

## 2. 三个丢失点

| # | 位置 | 丢失内容 |
|---|---|---|
| 1 | [ray_trainer_ts.py:843-846](../verl/trainer/ppo/ray_trainer_ts.py#L843-L846) | `batch.pop` 只保留 `uid/data_source/reward_model`，`extra_info` 里的证据字段不进 `gen_batch` |
| 2 | [generation_ts.py:855-870](../search_r1/llm_agent/generation_ts.py#L855-L870) | `_passages2string` 只用 `contents` 拼成 `Doc N(Title: ...) ...`，Wiki 返回的 `document.id` 被丢弃 |
| 3 | [main_ppo_format_ts.py:44-98](../verl/trainer/main_ppo_format_ts.py#L44-L98) | `RewardManager.__call__` 只读 `reward_model` / `data_source` 计算答案+格式分，没有证据项 |

## 3. 关键不变量：叶子顺序与 final_output 行序对齐

接线正确性的前提是「叶子顺序 == final_output 行序」。该对齐由两层机制共同保证：

- `final_node_list` 按 root 分组：[generation_ts.py:629-634](../search_r1/llm_agent/generation_ts.py#L629-L634)
  对每个 root `sample_leaf(k)` 后 `extend`，得到 `[root0 的 k 叶, root1 的 k 叶, ...]`；
  [tree_node.py:145-173](../search_r1/llm_agent/tree_node.py#L145-L173) 的 `sample_leaf`
  返回 `get_subtree_nodes` 的 BFS 顺序叶子（与 `random.shuffle` 无关）。
- `final_output = gen_batch.repeat(k, interleave=True)` 用 `np.repeat`，行序为
  `[根样本0 ×k, 根样本1 ×k, ...]`。

因此只要证据字段随 `gen_batch.repeat(k)` 传播，其第 `i` 行就与 `final_node_list[i]`
所属的根样本一一对应（`i // k` 定位根样本）。`final_output.batch['responses']` 更是
直接由 `final_node_list` 的 `node.responses` 拼成
（[generation_ts.py:636-642](../search_r1/llm_agent/generation_ts.py#L636-L642)），
对应关系由构造保证。**已由 `tests/researchsearch/test_tree_evidence_wiring.py`
的 `test_leaf_order_aligns_with_interleaved_repeat` 锁定。**

## 4. 接线方案（Day 6，待 GPU 验证）

按最小侵入改三处，保持协议与 Agent 观察字符串不变：

### 4.1 让证据字段随 gen_batch 传入（丢失点 1）

在 [ray_trainer_ts.py:843-846](../verl/trainer/ppo/ray_trainer_ts.py#L843-L846) 把
`extra_info` 加入 `non_tensor_batch_keys`（或在其前把 `sample_id` /
`supporting_doc_ids` / `evidence_status` / `corpus_version` 从 `extra_info` 提取为独立
object 数组再 pop）。推荐后者：只传奖励回调需要的窄字段，避免整列 dict 随树展开重复。

```python
non_tensor_batch_keys = ['uid', 'data_source', 'reward_model',
                         'sample_id', 'supporting_doc_ids', 'evidence_status']
```

### 4.2 旁路保留文档 ID 并逐叶收集（丢失点 2）

- `_batch_search` 已返回原始 `{'result': [[...]]}`；`batch_search` 目前只取
  `result` 交给 `_passages2string` 拼字符串。改为同时用已交付的
  `evidence.extract_doc_ids_from_retrieval` 提取每个查询的 `document.id`，随
  `execute_predictions` 的返回一并带回，记到该 step 生成的 `TreeNode` 上（新字段
  `visible_doc_ids`）。
- 「截断后可见」的语义：`_process_next_obs` 会把 observation 截断到
  `max_obs_length`（token 级）。因此须按截断后的 observation 判断哪些文档仍完整保留，
  被截掉的文档记入 `retrieval_hit_doc_ids`，不计入 `R_evidence` 分子
  （方案 §4.4）。这一判定依赖文档在格式化字符串中的边界，是 Day 6 接线最细的部分，
  不能只用「后端返回了几个」替代。
- `run_llm_loop_tree_search` 在收集 `final_node_list` 后，按叶子顺序把每个叶子的
  `visible_doc_ids` 挂到 `final_output.non_tensor_batch['visible_doc_ids']`（object
  数组），与 `final_node_list` 一一对应。

### 4.3 奖励回调计算证据项（丢失点 3）

[main_ppo_format_ts.py](../verl/trainer/main_ppo_format_ts.py) 的 `RewardManager`
增加 `evidence_weight`（默认 0），在逐条循环里读
`data_item.non_tensor_batch['supporting_doc_ids']` / `['evidence_status']` /
`['visible_doc_ids']`，调用已交付的 `evidence.compute_evidence_reward`，把
`alpha * score` 加到该条 `score`，并把 `eligible` / `num_covered` / `num_gold` 汇入
日志。`alpha=0` 为默认，`alpha=0.1` 仅作接线验证值。

## 5. 已交付的预备组件

| 组件 | 位置 | 覆盖 |
|---|---|---|
| doc_id 提取纯函数 | `verl/utils/reward_score/evidence.py::extract_doc_ids_from_retrieval` | 丢失点 2 的旁路提取，保序、跳过缺失 |
| 叶子顺序对齐测试 | `tests/researchsearch/test_tree_evidence_wiring.py` | `sample_leaf` BFS 顺序、`np.repeat` 对齐、doc_id 提取、两题不串题（6 项） |
| 优势路由测试 | `tests/researchsearch/test_adv_routing.py` | `compute_advantage` 的 `tree`/`tree_2norm` 路由（4 项） |

## 6. 尚待 GPU 验证（未伪报）

- 真实 rollout 中 `sample_id` / `supporting_doc_ids` / `evidence_status` 确实随
  `gen_batch.repeat` 到达树内 `reward_fn`。
- 逐叶 `visible_doc_ids` 与 `final_node_list` 一一对应，且两题证据不同时预期分数改变、
  不串题（方案 §6 Day 6 任务 1 验收）。
- 截断后可见性与 `retrieval_hit` 的判定在真实 observation 上正确。
