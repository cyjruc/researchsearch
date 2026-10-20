# loss mask 静态分析（门禁 T06 前置）

对齐《第 1 周工程闭环开发方案》T06：工具观察和 padding 不计策略损失，模型生成正确参与。本文记录 `info_mask → loss_mask` 的完整链路与当前结论，基于源码静态核查；运行时断言（需一次真实 rollout）另列于末尾。

## 1. 两个 mask 的定义

`tensor_helper.py:35` 的 `create_attention_mask` 语义为 `(input_ids != pad_token_id)`，即非 pad 的 token 为 1，pad 为 0。

| mask | 构造来源 | 语义 |
|---|---|---|
| `attention_mask` | `create_attention_mask(真实 token)` | prompt、agent 生成 token、`<information>` 观察 token 均为 1；padding 为 0 |
| `info_mask` | `create_attention_mask(responses_with_info_mask)` | `<information>` 观察 token 为 0；其它非 pad 为 1；padding 为 0 |

关键：`responses_with_info_mask` 与真实 `responses` 完全相同，只是把 `<information>` block 的每个 token 替换为 `pad_token_id`（`generation.py:135`、`generation_ts.py:161`：`info_mask = torch.full(info.size(), pad_id)`）。因此对 `responses_with_info_mask` 求 attention mask 时，information 位置因等于 pad 而自动归 0——这是"用 pad 占位"实现的观察掩码，不是显式位置掩码。

## 2. 链路（chain 与 tree 路径一致）

1. **生成**：`_info_masked_concatenate_with_padding` 同步构造 `responses`（真实）与 `responses_with_info_mask`（information→pad），随后按 padding 排序并截断。
2. **组装**（`generation.py:354-357` / `generation_ts.py:655-658`）：
   ```python
   info_mask = attention_mask(prompts) ⊕ attention_mask(responses_with_info_mask)
   ```
3. **派生**（`ray_trainer.py:879-892` / `ray_trainer_ts.py:1003-1016`，两个 `_create_loss_mask` 完全一致）：
   ```python
   response_length = responses.shape[-1]
   response_mask = attention_mask[:, -response_length:]
   loss_mask = info_mask[:, -response_length:]   # 仅取 response 段
   ```
4. **消费**（`dp_actor.py:280-282`）：
   ```python
   response_mask = attention_mask[:, -response_length:]
   if state_masking:
       response_mask = data['loss_mask']        # 用 info_mask 派生的 loss_mask 覆盖
   ```
5. **聚合**（`core_algos.py` 的 `compute_policy_loss_*` → `agg_loss`）：`token-mean`（默认）等模式都对 `loss_mat * loss_mask` 求损失，information 与 padding 位置乘 0。

## 3. T06 验证结论

| 要求 | 结论 | 依据 |
|---|---|---|
| 工具观察不计策略损失 | ✅（`state_masking=true` 时） | info_mask 在 information 位置=0 → loss_mask=0 → agg_loss 乘 0 |
| padding 不计 | ✅ | attention_mask=0 → info_mask=0 → loss_mask=0 |
| 模型生成正确参与 | ✅ | think/search/answer token 的 info_mask=1 |

**前提开关**：`actor_rollout_ref.actor.state_masking=true`。config 默认值为 `False`（`ppo_trainer.yaml:36`），但所有 `train_*.sh` 与 `scripts/researchsearch/launch_smoke.sh` 均覆盖为 `true`（如 `train_multihopqa_tree_search.sh:79`）。若误用默认值，`response_mask` 退化为 `attention_mask`，`<information>` 观察将参与策略损失，违反 T06。新增脚本须显式带上该开关。

## 4. 树路径特有

- `TreeNode.responses_with_info_mask` 与 `TreeNode.turns_mask`（`tree_node.py:31-32,56-57`）随节点保存，共享前缀的这两个 mask 一并复用，分支不重算公共部分。
- 叶子顺序：遍历 `root_list` → `root.sample_leaf(k)`，`final_node_list` 顺序为 **BS × M(树) × K(叶子)**（`generation_ts.py:629-639`），与 `final_responses` 逐行对应。
- `final_output.batch['info_mask'] = attention_mask(prompts) ⊕ attention_mask(pad_and_stack(叶子 responses_with_info_mask))`（`generation_ts.py:655-658`），与 chain 路径语义一致。

## 5. 三个 mask 的关系

| mask | 用途 | 是否随树传播 |
|---|---|---|
| `attention_mask` | 有效 token（用于 KL/优势的 `response_mask` 底版、position） | 由叶子 `responses` 重建 |
| `info_mask` → `loss_mask` | 是否计入策略损失（mask 掉 observation 与 padding） | 由叶子 `responses_with_info_mask` 重建 |
| `turns_mask` | 每个 token 属于第几轮（供 `policy_loss='turn'` 做轮次级 ratio） | 随叶子 `turns_mask` 传播 |

`turns_mask` 与 `loss_mask` 正交：前者决定 token 属于哪个 turn，后者决定 token 是否计入损失。当前训练脚本显式 `policy_loss=grpo`（`train_*.sh`），`compute_policy_loss_turn` 未被使用，`turns_mask` 只生成不消费；只有切到 `policy_loss=turn` 时才启用。

## 6. 已知边界与风险（如实记录）

1. **信息掩码靠 pad 占位**：information 内容 token 若恰好等于 `pad_token_id`，会被误判为 observation 位置而被 mask。正常语料不命中（pad 是独立特殊 token），但这是"替换为 pad"而非显式位置掩码的固有边界。
2. **`state_masking` 是开关**：默认 False，靠脚本覆盖 true。新配置若遗漏，会静默让观察参与损失而不报错。
3. **截断对齐**：`_update_right_side` 将 `responses` / `responses_with_info_mask` / `turns_mask` 一起截断（`generation_ts.py:207-214`），超长轨迹截断后 `info_mask` 仍与 `responses` 逐位置对齐。
4. **`turns_mask` 的 dtype 与 info_mask 混用**：`generation_ts.py:163` 把 `info_mask`（值为 pad_id 的全量张量）append 进 `turns_mask_tensors`，information 段的 turns_mask 由此为 0，语义上等价于"观察不属于任何轮次"；但这是隐式约定，改动 turns 逻辑时需注意。

## 7. 待执行运行时验收（需真实 rollout，本机无 GPU 未执行）

对一条含 `<information>` 的轨迹断言：`loss_mask` 在 information 区间全 0、在 think/search/answer 区间全 1、padding 全 0，且与 `attention_mask` 只在 observation 区间有差异。同时核验树路径 `final_node_list` 顺序与 `loss_mask` 逐行对应。
