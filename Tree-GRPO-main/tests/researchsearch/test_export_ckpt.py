# Copyright 2026 ResearchSearch-RL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""checkpoint 导出脚本测试（方案 §6 Day 6 任务 2 无卡部分 / 阻塞 4）。

锁定 ``scripts/researchsearch/export_ckpt.sh`` 的参数化行为：缺参数提前失败、
``--dry-run`` 不真正合并只解析路径、合并器通过脚本自身目录定位（不依赖工作目录）。
真实合并需要 FSDP checkpoint 与 torch 分布式依赖，属 Day 6 GPU 阶段，此处不执行。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # Tree-GRPO-main
_SCRIPT = _REPO / "scripts/researchsearch/export_ckpt.sh"
_MERGER = _REPO / "scripts/merge_ckpt/model_merger.py"


def _run(*args, extra_env=None):
    # 从干净环境启动，避免继承调用方 shell 里可能存在的 LOCAL_DIR/TARGET_DIR。
    env = {"PATH": os.environ.get("PATH", "")}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(_SCRIPT), *args], capture_output=True, text=True, env=env
    )


def test_missing_required_paths_fail_early():
    result = _run("--dry-run")
    assert result.returncode == 2
    assert "LOCAL_DIR" in result.stderr
    assert "TARGET_DIR" in result.stderr


def test_dry_run_resolves_paths_without_merging():
    result = _run(
        "--dry-run",
        extra_env={"LOCAL_DIR": "/tmp/ckpt/global_step_2/actor", "TARGET_DIR": "/tmp/merged_hf"},
    )
    assert result.returncode == 0
    # 解析后的命令包含参数化输入输出与合并器路径。
    assert "/tmp/ckpt/global_step_2/actor" in result.stdout
    assert "/tmp/merged_hf" in result.stdout
    assert "model_merger.py" in result.stdout
    assert "No model was merged" in result.stdout


def test_hf_model_path_only_passed_when_set():
    with_path = _run(
        "--dry-run",
        extra_env={
            "LOCAL_DIR": "/tmp/a",
            "TARGET_DIR": "/tmp/b",
            "HF_MODEL_PATH": "/tmp/base",
        },
    )
    without_path = _run(
        "--dry-run",
        extra_env={"LOCAL_DIR": "/tmp/a", "TARGET_DIR": "/tmp/b"},
    )
    # printf '%q' 会把各参数分别转义并用空白分隔，故只断言参数名与值分别出现。
    assert "--hf_model_path" in with_path.stdout
    assert "/tmp/base" in with_path.stdout
    assert "--hf_model_path" not in without_path.stdout


def test_invalid_action_fails():
    result = _run("--bogus")
    assert result.returncode == 2


def test_merger_is_located_relative_to_script():
    # 脚本以自身目录定位合并器，合并器文件必须存在。
    assert _MERGER.is_file()
