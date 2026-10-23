# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`Tree-GRPO-main/` is a fork of [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO) (ICLR 2026), itself built on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [veRL](https://github.com/verl-project/verl). It RL-trains a ReAct-style LLM agent, contrasting two rollout strategies: independent chain rollouts (GRPO) vs. tree-search rollouts (Tree-GRPO), which sample over a search tree of ReAct step-level nodes to spend less rollout budget and yield a tree-based process-supervision signal.

On top of the upstream code, this workspace runs a two-stage "ResearchSearch-RL" project: frozen-corpus Wiki RL → web-search RL, plus a doc-level evidence-coverage reward. The authoritative project plans live at the workspace root (one level up) and take precedence over the upstream README:

- `../ResearchSearch-RL-项目架构与实验设计.md` — architecture, the five experiments E0–E4, reward/evidence definitions, metrics.
- `../ResearchSearch-RL-第1周工程闭环开发方案.md` — week-1 source-level plan, known gaps, acceptance gates T01–T10.
- `../AGENTS.md` — repository commands, structure, style, and hygiene conventions (loaded alongside this file).

All paths below are relative to `Tree-GRPO-main/`.

## Environments and commands

Two separate conda envs (README "Installation"): `treegrpo` for RL (torch 2.6, vLLM 0.8.5, flash-attn, swanlab) and `retriever` for search (pyserini, faiss-gpu, uvicorn/fastapi). Training additionally needs a Ray cluster, GPU host, model weights (default Qwen2.5-3B), datasets, and a running retrieval server.

```bash
pip install -e '.[test]'                          # editable install; [test] adds pytest + yapf
bash scripts/data_process/data_process_multihop.sh   # build multi-hop QA parquet
bash scripts/data_process/data_process_singlehop.sh  # build single-hop QA parquet
bash local_retrieval_launch.sh                    # local FAISS/E5 /retrieve server (retriever env)
bash bing_search_launch.sh                        # Bing/Brightdata search server (retriever env)
bash train_multihopqa_grpo.sh                     # chain-based GRPO baseline
bash train_multihopqa_tree_search.sh              # Tree-GRPO
python -m pytest tests/                           # run tests (directory is planned, not yet present)
python -m pytest tests/test_x.py::test_name       # run one test
```

The launcher scripts ship with empty/hardcoded `DATA_DIR`/model paths and an undefined `$lr`; parameterize and preflight before submitting. Week-1 helper scripts under `scripts/researchsearch/` (`preflight.py`, `ray_worker_probe.py`, `launch_smoke.sh`, `launch_search_service.sh`) exist for this purpose.

## Architecture

### Two parallel training stacks

Both share the veRL PPO scaffold but differ in entry point, trainer, and rollout:

| | Chain-based GRPO | Tree-GRPO |
|---|---|---|
| Entry | `verl/trainer/main_ppo_format.py` | `verl/trainer/main_ppo_format_ts.py` |
| Trainer | `verl/trainer/ppo/ray_trainer.py` | `verl/trainer/ppo/ray_trainer_ts.py` |
| Rollout | `search_r1/llm_agent/generation.py` (`LLMGenerationManager`) | `search_r1/llm_agent/generation_ts.py` (`LLMGenerationTreeSearchManager.run_llm_loop_tree_search`) |
| Nodes | flat sequences | `search_r1/llm_agent/tree_node.py` (`TreeNode`) |

Hydra config is in `verl/trainer/config/` (`ppo_trainer.yaml` is the main one).

### ReAct protocol

Agent output uses `<think>`, `<search>`, `<information>`, `<answer>` tags. The format state machine lives in `verl/utils/reward_score/qa_em_format.py` and `qa_f1_format.py` (`is_valid_sequence`), enforcing a valid think→search→information→…→answer ordering.

### Tree-search rollout (the core idea)

`run_llm_loop_tree_search` expands a tree at **agent-step boundaries** — a node is a prefix through one complete ReAct step. Shared prefixes are stored once and reused (token ids, masks), so sibling branches don't re-generate the common part. Per tree, `ts_k` leaves are sampled (`root.sample_leaf(k)`); the final output batch is assembled from `final_node_list`, and rewards are computed on those leaves **inside** tree sampling, before merging with the original batch. Tree-shape params `ts_m`/`ts_n`/`ts_l`/`ts_k` correspond to M/N/L/K in the launch scripts; K is the number of retained leaves per tree, not the number of independent trajectories or the physical token cost.

### Reward and advantage

- **Reward**: `RewardManager` in `main_ppo_format(_ts).py` wraps `compute_score_em`/`compute_score_f1`. Scoring is segmented: correct + valid format = 1; correct + invalid format loses the structure score; wrong + valid format may earn structure score. `retrieval_score` only checks whether `<information>` contains the answer string — this project fixes it to 0. An `R_evidence` term (doc-level gold-evidence coverage, `R_total = R_existing + evidence_alpha * R_evidence`) is wired into the tree trainer's `RewardManager` (default `evidence_alpha: 0.0` = off). Evidence fields flow through `gen_batch`'s `extra_info` plus per-leaf `visible_doc_ids` into the in-tree `reward_fn`; see `docs/researchsearch/evidence-wiring.md`.
- **Advantage**: in `verl/trainer/ppo/core_algos.py`. `compute_grpo_outcome_advantage` normalizes per question; `compute_tree_grpo_outcome_advantage` normalizes twice — per question `index`, then per tree `tree_uid`. `compute_advantage` in `ray_trainer(_ts).py` routes on `adv_estimator` (`grpo` vs `tree`). `adv_estimator=tree` uses the double-normalization path, not the `tree_2norm`-routed function mentioned in some comments.

### Retrieval backends

- `search_r1/search/retrieval_server.py` — local FAISS/E5 `/retrieve` (dense + sparse), returns `{"result":[[{"document":{"id","contents"},"score"}]]}`.
- `search_r1/search/bing_search_server_redis.py` — web search (Brightdata) with Redis cache + periodic/per-termination JSON persistence; returns formatted string summaries.
- Rollout request shape: `{"queries":[...], "topk":3, "return_scores":true}`. Wiki returns structured docs (stable doc id + score); Web returns summary strings. Keep the two return shapes distinct in tooling and logging.

### Data flow and masking

Data is parquet with `data_source`, `prompt`, `reward_model` (holding `ground_truth.target`), and `extra_info`; loaded by `verl/utils/dataset/rl_dataset.py`. Two masks matter: `info_mask` (which tokens are agent-visible) and `loss_mask` (which tokens contribute to policy loss). Tool `<information>` observations and padding must be masked out of the policy loss.

## Known pitfalls (don't re-discover these)

1. **`index` collision**: `scripts/data_process/qa_search_train_merge.py` regenerates `index` from 0 for each dataset before concatenating, and both trainers use `index` as the advantage-grouping `uid`, so multi-source batches mix questions. Assign a global unique integer `index` per question; keep the raw row number as `source_index`.
2. **Evidence doesn't reach the tree reward**: the tree trainer `pop`s non-tensor fields before generation and tree sampling calls the reward callback early, so fields added only to parquet `extra_info` (e.g. `supporting_doc_ids`) are invisible to the in-tree reward unless carried through the generation batch. This is now wired: `ray_trainer_ts.py` conditionally carries `extra_info` into `gen_batch`, and `run_llm_loop_tree_search` writes per-leaf `visible_doc_ids` to `final_output` before `reward_fn`; see `docs/researchsearch/evidence-wiring.md`. Remaining gap: the Wiki retriever does not yet return a stable `document.id`, so real evidence coverage is empty until the corpus layer adds it (`docs/researchsearch/retrieval-contract.md` §2).
3. **`data.max_response_length` is per-turn**, not the whole multi-turn trajectory budget — track per-turn and total separately.
4. **Checkpoint ≠ resume**: FSDP save/load exists at the worker level (`verl/workers/fsdp_workers.py`, `verl/utils/checkpoint/fsdp_checkpoint_manager.py`) and stores model/optimizer/lr_scheduler/rng shards, but neither trainer wires any driver-side load — `fit` hard-codes `global_steps = 0` and never calls `load_checkpoint`. So only weight warm-start (export HF weights, point `model.path` at them, cold start) is reachable; exact resume (optimizer/scheduler/step/data-progress) is not. See `docs/researchsearch/checkpoint-resume.md`. Export is parameterized via `scripts/researchsearch/export_ckpt.sh` (`--merge` / `--dry-run`).
5. **No `.git`** in this snapshot — record a source snapshot/version hash rather than assuming upstream commit provenance.
