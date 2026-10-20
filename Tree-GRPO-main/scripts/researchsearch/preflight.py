#!/usr/bin/env python3
"""Preflight checks for a ResearchSearch-RL smoke run."""

import argparse
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def report_local_environment():
    print("Local environment (credential values are never printed):")
    print("  platform:", platform.platform())
    print("  machine:", platform.machine())
    print("  python:", sys.version.split()[0], sys.executable)
    for executable in ("conda", "uv", "nvidia-smi", "nvcc", "ray"):
        print("  {}: {}".format(executable, shutil.which(executable) or "not found"))
    for module in ("torch", "ray", "vllm", "transformers", "datasets"):
        available = importlib.util.find_spec(module) is not None
        print("  python module {}: {}".format(module, "available" if available else "not installed"))
    if not shutil.which("nvidia-smi"):
        print("  CUDA GPU: unavailable on this host")
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10,
        )
        print("  nvidia-smi:", result.stdout.strip() or "no device information")


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError("required environment variable is missing: {}".format(name))
    return value


def validate_launch_config(create_log_dir):
    errors = []
    values = {}
    for name in ("DATA_DIR", "MODEL_PATH", "LEARNING_RATE", "N_GPUS_PER_NODE", "LOG_DIR", "RAY_ADDRESS", "RETRIEVER_URL"):
        try:
            values[name] = required_env(name)
        except ValueError as error:
            errors.append(str(error))

    if "DATA_DIR" in values:
        data_dir = Path(values["DATA_DIR"]).expanduser()
        train_file = Path(os.environ.get("TRAIN_FILE", str(data_dir / "train.parquet"))).expanduser()
        dev_file = Path(os.environ.get("DEV_FILE", str(data_dir / "dev.parquet"))).expanduser()
        if not data_dir.is_dir():
            errors.append("DATA_DIR is not a directory")
        if not train_file.is_file():
            errors.append("training parquet is missing: {}".format(train_file))
        if not dev_file.is_file():
            errors.append("validation parquet is missing: {}".format(dev_file))
        values["TRAIN_FILE"] = str(train_file.resolve())
        values["DEV_FILE"] = str(dev_file.resolve())

    if "MODEL_PATH" in values:
        model_path = Path(values["MODEL_PATH"]).expanduser()
        if not model_path.is_dir():
            errors.append("MODEL_PATH must point to an existing local model directory")
        elif not (model_path / "config.json").is_file():
            errors.append("MODEL_PATH is missing config.json")
        values["MODEL_PATH"] = str(model_path.resolve())

    if "LEARNING_RATE" in values:
        try:
            rate = float(values["LEARNING_RATE"])
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError
        except ValueError:
            errors.append("LEARNING_RATE must be a finite positive number")

    if "N_GPUS_PER_NODE" in values:
        try:
            if int(values["N_GPUS_PER_NODE"]) < 1:
                raise ValueError
        except ValueError:
            errors.append("N_GPUS_PER_NODE must be a positive integer")

    for name in ("RAY_ADDRESS", "RETRIEVER_URL"):
        if name in values and not values[name].startswith(("http://", "https://")):
            errors.append("{} must be an http(s) URL".format(name))

    if "LOG_DIR" in values:
        log_dir = Path(values["LOG_DIR"]).expanduser()
        if create_log_dir:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                errors.append("cannot create LOG_DIR: {}".format(error))
        elif log_dir.exists() and not log_dir.is_dir():
            errors.append("LOG_DIR exists and is not a directory")
        values["LOG_DIR"] = str(log_dir.resolve())

    if errors:
        for error in errors:
            print("ERROR:", error, file=sys.stderr)
        return None

    print("Launch configuration checks passed:")
    for name in ("DATA_DIR", "TRAIN_FILE", "DEV_FILE", "MODEL_PATH", "N_GPUS_PER_NODE", "LOG_DIR", "RAY_ADDRESS", "RETRIEVER_URL"):
        print("  {}: {}".format(name, values.get(name, "configured")))
    print("  LEARNING_RATE: configured")
    return values


def check_runtime(gpu_count):
    missing = [name for name in ("torch", "ray", "vllm", "transformers") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("required runtime packages are missing: {}".format(", ".join(missing)))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run training preflight on a compatible Linux GPU host")
    if torch.cuda.device_count() < gpu_count:
        raise RuntimeError("requested {} GPUs, but PyTorch sees {}".format(gpu_count, torch.cuda.device_count()))
    print("CUDA runtime checks passed: {} visible GPU(s)".format(torch.cuda.device_count()))


def check_ray(address):
    executable = shutil.which("ray")
    if not executable:
        raise RuntimeError("ray CLI is not installed in the active training environment")
    result = subprocess.run(
        [executable, "job", "list", "--address", address], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20,
    )
    if result.returncode:
        raise RuntimeError("Ray job endpoint check failed: {}".format(result.stdout.strip()))
    print("Ray job endpoint is reachable")


def check_ray_worker(address, data_dir, model_path, retriever_url):
    executable = shutil.which("ray")
    if not executable:
        raise RuntimeError("ray CLI is not installed in the active training environment")
    command = [
        executable, "job", "submit", "--address", address,
        "--runtime-env", "verl/trainer/runtime_env.yaml", "--",
        "python3", "scripts/researchsearch/ray_worker_probe.py",
        "--data-dir", data_dir,
        "--model-path", model_path,
        "--retriever-url", retriever_url,
    ]
    result = subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=150,
    )
    if result.returncode:
        raise RuntimeError("Ray worker probe failed: {}".format(result.stdout.strip()))
    print("Ray worker probe passed:")
    print(result.stdout.strip())


def probe_retriever(url):
    payload = json.dumps({"queries": ["ResearchSearch smoke connectivity check"], "topk": 3, "return_scores": True}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            body = json.loads(response.read().decode("utf-8"))
            if response.status < 200 or response.status >= 300:
                raise RuntimeError("retriever returned HTTP {}".format(response.status))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("retriever probe failed: {}".format(error))
    results = body.get("result") if isinstance(body, dict) else None
    if not isinstance(results, list):
        raise RuntimeError("retriever response is missing a result list")
    print("Retriever probe passed: HTTP 200, {} query result(s)".format(len(results)))


def run_model_smoke(model_path, learning_rate):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("model generation and optimizer smoke requires CUDA on the target training host")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", local_files_only=True,
    )
    model.eval()
    prompt = tokenizer("Answer briefly: 2 + 2 =", return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    input_ids = prompt["input_ids"].to(input_device)
    attention_mask = prompt["attention_mask"].to(input_device)
    with torch.no_grad():
        generated = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=8)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
    if not torch.isfinite(loss):
        raise RuntimeError("model smoke produced a non-finite loss")
    loss.backward()
    optimizer.step()
    print("Model smoke passed: generated {} token(s), optimizer loss {:.6f}".format(
        generated.shape[-1] - input_ids.shape[-1], loss.detach().float().item()
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-local", action="store_true", help="report machine information without launch settings")
    parser.add_argument("--for-launch", action="store_true", help="validate launch variables and local paths")
    parser.add_argument("--check-runtime", action="store_true", help="require PyTorch, Ray, vLLM and CUDA")
    parser.add_argument("--check-ray", action="store_true", help="query the configured Ray Jobs endpoint")
    parser.add_argument("--check-ray-worker", action="store_true", help="verify runtime_env, GPU, shared paths and retriever connectivity on a worker")
    parser.add_argument("--probe-retriever", action="store_true", help="send one real request to RETRIEVER_URL")
    parser.add_argument("--model-smoke", action="store_true", help="run a short generation and one AdamW update")
    args = parser.parse_args()
    if args.report_local:
        report_local_environment()
    needs_launch = any((args.for_launch, args.check_runtime, args.check_ray, args.check_ray_worker, args.probe_retriever, args.model_smoke))
    values = validate_launch_config(create_log_dir=args.for_launch) if needs_launch else None
    if needs_launch and values is None:
        return 2
    try:
        gpu_count = int(values["N_GPUS_PER_NODE"]) if values else 0
        if args.check_runtime or args.model_smoke:
            check_runtime(gpu_count)
        if args.check_ray:
            check_ray(values["RAY_ADDRESS"])
        if args.check_ray_worker:
            check_ray_worker(values["RAY_ADDRESS"], values["DATA_DIR"], values["MODEL_PATH"], values["RETRIEVER_URL"])
        if args.probe_retriever:
            probe_retriever(values["RETRIEVER_URL"])
        if args.model_smoke:
            run_model_smoke(values["MODEL_PATH"], float(os.environ["LEARNING_RATE"]))
    except (RuntimeError, OSError, ValueError) as error:
        print("ERROR:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
