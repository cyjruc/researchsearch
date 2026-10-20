# Repository Guidelines

## Project Structure & Module Organization

The project lives in `Tree-GRPO-main/`; run the commands below from that directory.

- `search_r1/llm_agent/`: agent generation, tree-search rollouts, tree nodes, and tensor helpers.
- `search_r1/search/`: retrieval and web-search servers, plus cache utilities.
- `verl/`: reinforcement-learning trainers, distributed workers, models, and shared utilities. Hydra configuration lives in `verl/trainer/config/`.
- `scripts/data_process/`: dataset preparation; `scripts/merge_ckpt/`: checkpoint conversion.
- `resources/`: README images. Root-level `train_*.sh` files launch training experiments.

## Build, Test, and Development Commands

Follow `README.md` to create separate training and retriever environments with the documented PyTorch, vLLM, and FAISS versions.

- `pip install -e '.[test]'`: install the local package in editable mode with pytest and YAPF.
- `bash scripts/data_process/data_process_multihop.sh` or `bash scripts/data_process/data_process_singlehop.sh`: prepare QA datasets.
- `bash local_retrieval_launch.sh`: start local retrieval after configuring index, corpus, and model paths.
- `bash train_multihopqa_grpo.sh`: submit baseline GRPO training.
- `bash train_multihopqa_tree_search.sh`: submit tree-search training.

Training requires configured datasets, model weights, retrieval services, GPUs, and a running Ray cluster. Inspect launcher settings before execution; the multi-hop tree-search launcher defaults to eight GPUs.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` for functions and modules, and `PascalCase` for classes. Preserve surrounding style and existing license headers. Add type annotations where they clarify tensor or configuration interfaces. YAPF is available through the test extra, but no formatter configuration is included; avoid unrelated formatting changes. Follow the existing `train_<task>_<method>.sh` naming pattern.

## Testing Guidelines

This snapshot includes pytest as an optional dependency but no automated test suite or coverage threshold. Add focused regression tests under `Tree-GRPO-main/tests/`, named `test_*.py`, and run `python -m pytest tests/`. Mock external search services for unit tests. For training changes, document a small GPU smoke run, including configuration and observed results.

## Commit & Pull Request Guidelines

No Git metadata is included, so historical commit conventions cannot be verified. Use concise, imperative commit subjects such as `Fix tree rollout termination`. PRs should explain behavior changes, affected configurations, validation performed, and linked issues where applicable. Include experiment settings and metrics for algorithm changes.

## Configuration & Data Hygiene

Keep API credentials, downloaded corpora, indexes, model weights, checkpoints, and generated logs out of commits. Replace launcher placeholder paths locally and check dataset, GPU, and Ray settings before submitting jobs.
