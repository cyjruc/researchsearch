#!/usr/bin/env python3
"""Check that a Ray worker sees runtime dependencies, paths, and retriever."""

import argparse
import json
import socket
from urllib.parse import urlparse


def inspect_worker(data_dir, model_path, retriever_url):
    import importlib.util
    import os
    import platform
    import sys
    import torch
    import ray

    missing = [name for name in ("vllm", "transformers") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("worker is missing packages: {}".format(", ".join(missing)))

    from pathlib import Path

    data = Path(data_dir)
    model = Path(model_path)
    missing_paths = [
        str(path) for path in (
            data / "train.parquet", data / "dev.parquet", model / "config.json"
        ) if not path.is_file()
    ]
    if missing_paths:
        raise RuntimeError("worker cannot access required files: {}".format(", ".join(missing_paths)))

    parsed = urlparse(retriever_url)
    if not parsed.hostname or not parsed.port:
        raise RuntimeError("retriever URL must include a hostname and port")
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=5):
            pass
    except OSError as error:
        raise RuntimeError("worker cannot connect to retriever host: {}".format(error))

    runtime_env_checks = {
        "SWANLAB_MODE": os.environ.get("SWANLAB_MODE") == "cloud",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS") == "1",
    }
    if not all(runtime_env_checks.values()):
        raise RuntimeError("runtime_env.yaml environment variables did not reach the worker")
    if not torch.cuda.is_available():
        raise RuntimeError("Ray worker has no visible CUDA device")

    return {
        "node": platform.node(),
        "python": sys.version.split()[0],
        "ray": ray.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "data_files_accessible": True,
        "model_config_accessible": True,
        "retriever_tcp_reachable": True,
        "runtime_env_checks": runtime_env_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--retriever-url", required=True)
    args = parser.parse_args()

    import ray

    ray.init(address="auto", logging_level="ERROR")
    try:
        remote_inspector = ray.remote(num_gpus=1)(inspect_worker)
        result = ray.get(
            remote_inspector.remote(args.data_dir, args.model_path, args.retriever_url),
            timeout=90,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
