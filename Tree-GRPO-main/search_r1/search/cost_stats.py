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
"""统一成本统计口径（架构 §3 / 第 2 周方案 §6.3、T10）。

权威口径（架构 §3）：分别报告物理新生成 token、逻辑搜索次数、实际外部请求
（含重试）、GPU 时间；共享前缀不重复计入真实生成成本。第 2 周方案 §6.3 追加：
缓存命中单独计，checkpoint 体积与磁盘占用写入日志（第 1 周单卡 FSDP checkpoint
≈36G/step 在 8-GPU 分片下不复现，但仍须记录体积）。

本模块提供：

- ``COST_FIELDS``：六口径的规范字段名（单一权威定义，Wiki / Web / 训练三侧共用）。
- ``CostStats``：进程内线程安全的累加器；跨 Ray worker 用 ``merge`` 聚合。
- ``physical_new_tokens_from_tree``：遍历树节点累加「模型生成、排除 information 与
  padding」的物理新生成 token（共享前缀只计一次，因节点存的是增量 responses）。
- ``directory_size_bytes``：递归统计目录体积，用于 checkpoint 磁盘占用。

与已落地代码的映射（Web 侧 ``get_stats()`` 三口径不重构，聚合时按此换算）：

- ``logical_searches``（动作粒度）～ Web 侧 ``logical_queries``（query 粒度）：
  一次 ``/retrieve`` 端点请求 = 1 次逻辑搜索，可含多个 query。
- ``http_requests``（含重试）== Web 侧 ``api_requests``。
- ``cache_hits`` == Web 侧 ``cache_hits``。

本模块零重依赖（不 import torch / search_r1 内部），``physical_new_tokens_from_tree``
对节点 tensor 走鸭子类型访问，便于在无 GPU 的本地单测里用 fake 节点锁定口径。
"""

from __future__ import annotations

import os
import threading
from typing import Dict

# 六口径规范字段名（架构 §3 / 第 2 周方案 §6.3 的单一权威定义）。
COST_FIELDS = (
    "physical_new_tokens",  # 物理新生成 token（共享前缀不重复计）
    "logical_searches",     # 逻辑搜索次数（Agent 搜索动作粒度）
    "http_requests",        # 实际外部请求（含重试）
    "cache_hits",           # 缓存命中
    "gpu_time_seconds",     # GPU 时间（秒）
    "checkpoint_bytes",     # checkpoint 体积（字节）
)


class CostStats:
    """进程内线程安全的成本累加器（六口径）。

    计数类字段（物理 token、逻辑搜索、外部请求、缓存命中）为整数；GPU 时间与
    checkpoint 体积为数值累加。跨 Ray worker / 进程用 :meth:`merge` 聚合后上报。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, float] = {field: 0 for field in COST_FIELDS}

    # -- 计数类（整数） ------------------------------------------------------

    def logical_search(self, n: int = 1) -> None:
        """记 1 次逻辑搜索（Agent 的一次检索动作，可含多个 query）。"""
        with self._lock:
            self._counts["logical_searches"] += n

    def http_request(self, n: int = 1) -> None:
        """记 1 次实际外部 HTTP 请求（含重试；每次网络尝试都计）。"""
        with self._lock:
            self._counts["http_requests"] += n

    def cache_hit(self, n: int = 1) -> None:
        with self._lock:
            self._counts["cache_hits"] += n

    def add_physical_new_tokens(self, n: int) -> None:
        with self._lock:
            self._counts["physical_new_tokens"] += int(n)

    # -- 数值累加类 ---------------------------------------------------------

    def add_gpu_seconds(self, seconds: float) -> None:
        with self._lock:
            self._counts["gpu_time_seconds"] += float(seconds)

    def add_checkpoint_bytes(self, nbytes: int) -> None:
        with self._lock:
            self._counts["checkpoint_bytes"] += int(nbytes)

    # -- 读取 / 合并 --------------------------------------------------------

    def as_dict(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._counts)

    def merge(self, other: "CostStats") -> "CostStats":
        """把另一个累加器的六口径逐个累加进自身，返回自身（便于链式聚合）。"""
        for field, value in other.as_dict().items():
            with self._lock:
                self._counts[field] += value
        return self

    def reset(self) -> None:
        with self._lock:
            for field in COST_FIELDS:
                self._counts[field] = 0


def physical_new_tokens_from_tree(nodes, pad_token_id: int) -> int:
    """累加树上所有节点的「模型生成」物理新 token，共享前缀不重复计。

    树节点存的是**增量** ``responses_with_info_mask``——本节点从父状态新生成的那
    一段，工具返回的 ``<information>`` 位置已用 ``pad_token_id`` 掩掉。因此父节点
    的那一段只在其自身节点计一次，子节点不复生成，天然满足「共享前缀不计入物理
    新生成 token」的口径。

    参数 ``nodes`` 是 ``TreeNode`` 的可迭代（调用方传 ``root.get_subtree_nodes()``，
    根节点的 responses 通常为空、可一并传入）。纯函数：只访问节点的
    ``responses_with_info_mask`` 属性，不 import 树实现，便于离线单测。
    """
    total = 0
    for node in nodes:
        mask = getattr(node, "responses_with_info_mask", None)
        if mask is None:
            continue
        # 1-D 单条序列；若被 pad_and_stack 成 2-D，对所有维求和同样正确。
        total += int((mask != pad_token_id).sum())
    return total


def directory_size_bytes(path: str) -> int:
    """递归统计目录下所有常规文件的总字节数（checkpoint 磁盘占用）。"""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            filepath = os.path.join(dirpath, name)
            try:
                if os.path.isfile(filepath):
                    total += os.path.getsize(filepath)
            except OSError:
                continue
    return total
