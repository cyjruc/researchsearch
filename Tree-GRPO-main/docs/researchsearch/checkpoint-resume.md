# Checkpoint 保存 / 阶段迁移 / 断点恢复：支持范围调查

更新时间：2026-10-22

本文静态核对「保存、阶段迁移、断点恢复」三件事在源码里的实际支持程度，明确
区分「仅权重 warm start」与「精确续训」，不伪报。结论基于本次读取快照，定位时
同时使用文件名与函数名。

参照：《第 1 周工程闭环开发方案》§5.2；CLAUDE.md 已知陷阱 #4。

## 结论速览

| 能力 | 源码支持 | 本周可否达成 |
|---|---|---|
| 保存 | ✅ worker 侧完整（model/optimizer/lr_scheduler/rng 分片 + HF config/tokenizer） | 是 |
| 阶段迁移（warm start，仅权重） | ⚠️ 底层有 load，但 driver 侧**未接线**；只能靠外部导出 HF 模型 + 改 `model.path` 冷启动 | 是（外部脚本方式） |
| 精确续训（optimizer/scheduler/step/数据进度） | ❌ driver 侧完全未接线，`fit` 硬编码 `global_steps=0` | 否 |

## 1. 底层能力：worker 侧保存了什么

`ActorRolloutRefWorker.save_checkpoint`（[fsdp_workers.py:645](../verl/workers/fsdp_workers.py#L645)）
委托 `FSDPCheckpointManager.save_checkpoint`（[fsdp_checkpoint_manager.py:96](../verl/utils/checkpoint/fsdp_checkpoint_manager.py#L96)），
每个 rank 落盘：

- `model_world_size_{ws}_rank_{r}.pt` —— FSDP 分片模型权重
- `optim_world_size_{ws}_rank_{r}.pt` —— FSDP 分片优化器状态
- `extra_state_world_size_{ws}_rank_{r}.pt` —— 内含 `lr_scheduler` 与 `rng`（`get_rng_state()`）
- 仅 rank 0 额外写 HF `config.json` 与 tokenizer（[fsdp_checkpoint_manager.py:144-145](../verl/utils/checkpoint/fsdp_checkpoint_manager.py#L144-L145)）

`CriticWorker` 同构（[fsdp_workers.py:923](../verl/workers/fsdp_workers.py#L923)）。

`load_checkpoint`（[fsdp_checkpoint_manager.py:51-94](../verl/utils/checkpoint/fsdp_checkpoint_manager.py#L51-L94)）
对称地恢复 model 分片、optimizer 分片、lr_scheduler，并在 `extra_state` 含 `rng`
时恢复随机状态（`'rng'` 缺失时向后兼容跳过）。**因此 worker 侧「保存」与「加载」
本身都覆盖 optimizer/scheduler/rng，不是只存权重。**

## 2. driver 侧缺口（精确续训为什么不存在）

两个 trainer 的 `fit` 完全一致地缺少恢复接线：

1. **无任何 load 调用**：`ray_trainer_ts.py` / `ray_trainer.py` 全文件 grep 不到
   `load_checkpoint` / `load_model`。`fit` 开头不加载 actor/critic 分片。
2. **`global_steps` 硬编码重置**：[ray_trainer_ts.py:791](../verl/trainer/ppo/ray_trainer_ts.py#L791)
   `self.global_steps = 0`；[ray_trainer.py:699](../verl/trainer/ppo/ray_trainer.py#L699) 同。
   无论从哪个 checkpoint 出发，步数都从 0 起。
3. **无 epoch / 数据进度恢复**：训练循环 `for epoch ... for batch_dict in
   self.train_dataloader`（[ray_trainer_ts.py:831-832](../verl/trainer/ppo/ray_trainer_ts.py#L831-L832)）
   每次从头迭代 dataloader，没有 sampler 状态或「已消费 batch 数」的持久化与恢复。
4. **driver 状态不落盘**：`_save_checkpoint`（[ray_trainer_ts.py:750-762](../verl/trainer/ppo/ray_trainer_ts.py#L750-L762)）
   只调 worker 的 `save_checkpoint`，driver 侧的 `global_steps`、epoch 计数、RNG、
   dataloader 进度没有任何文件记录；它们仅出现在目录名 `global_step_{self.global_steps}`
   里，而该值每次 fit 都从 0 重新计，与真实训练步数会不一致。

综上：**「保存」是真的，「精确续训」没有 driver 侧接线。** 当前唯一的恢复路径是
外部导出 HF 权重后改 `model.path` 冷启动，这只能算 **warm start（仅权重，新优化器、
step 从 0、数据从头）**，不是精确续训。

## 3. 阶段迁移（warm start）的可行路径

本周 Wiki → Web 迁移按 §5.2 第 2 项，只需权重迁移、Stage 2 默认新建优化器：

1. `scripts/researchsearch/export_ckpt.sh --merge` 把 FSDP actor 分片合并为 HF 模型
   （参数化 + `--dry-run` 已交付，见 `test_export_ckpt.py`）。
2. Web 训练进程把 `actor_rollout_ref.model.path` 指向合并产物，正常冷启动。
3. 记录 parent checkpoint 路径与优化器重建策略，`reference` 初始化自 Stage-1 actor。

这与「精确续训」是两回事，报告里只能写「warm start」。

## 4. 若要接精确续训，driver 侧需补的最小清单

以下仅为范围记录，不在本周无卡范围内实现：

1. `fit` 开头按 `resume_path` 调用 `actor_rollout_wg.load_checkpoint(...)` /
   `critic_wg.load_checkpoint(...)`，恢复 worker 侧的 optimizer/scheduler/rng 分片。
2. driver 侧状态文件（`trainer_state.json`）：记录 `global_steps`、epoch、driver RNG、
   dataloader 进度（或 sampler state），保存时写入、恢复时读回，并让
   `self.global_steps` 从该值继续而非硬编码 0。
3. dataloader 需支持从上次停止的 batch 序号继续（`sampler.set_epoch` + skip 已消费
   batch，或持久化迭代器状态）。
4. 恢复后校验：训练 loss、LR、optimizer step 计数与断点前一致，方可称「精确续训」。

## 5. 本周验收口径

- 交付 checkpoint 清单时，明确「保存 = 完整分片」「阶段迁移 = 外部导出 warm start」，
  不把「能保存」或「能 warm start」写成「精确续训」。
- `global_step` 目录名仅作本次 run 内的相对标记，不代表跨 run 的绝对已完成更新数。
