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
"""检索客户端：统一超时、重试与错误分类。

对齐《第 1 周工程闭环开发方案》§4.3 / §6：单次请求 30 秒、最多重试 2 次、
指数退避；对不可重试的鉴权错误立即失败；区分真实零结果、限流、网络失败与
非法返回。rollout 侧 ``_batch_search`` 通过本模块请求 Wiki / Web 检索服务。

与 ``retrieval_server.py``（Wiki，FAISS/E5）和 ``bing_search_server_redis.py``
（Web，Brightdata）配套的客户端侧；两个后端都返回 ``{"result": [...]}``。
"""

from __future__ import annotations

import time

import requests

# 方案 §4.3 初始建议：单次请求 30 秒、最多重试 2 次、指数退避。
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF = 1.0

# 限流与服务端暂时不可用的状态码，可指数退避重试。
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RetrievalError(Exception):
    """检索失败，按 ``kind`` 分类（方案 §6）。"""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


def post_retrieve(
    url: str,
    payload: dict,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> dict:
    """POST 检索请求并返回含 ``result`` 的 JSON dict；失败抛 ``RetrievalError``。

    错误分类（``RetrievalError.kind``）：

    - ``auth``：401/403 鉴权失败，立即失败不重试。
    - ``rate_limit``：429 限流，指数退避重试。
    - ``server``：5xx 服务端暂时故障，指数退避重试。
    - ``network``：连接失败 / 超时，指数退避重试。
    - ``invalid``：其他非 2xx、非法 JSON、非 dict 或缺 ``result`` 字段，立即失败。

    真实零结果（HTTP 200 + ``result=[]``）不是错误，正常返回。
    """
    last_error: RetrievalError | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            last_error = RetrievalError("network", f"检索超时（>{timeout}s）: {exc}")
        except requests.exceptions.RequestException as exc:
            last_error = RetrievalError("network", f"检索连接失败: {exc}")
        else:
            status = response.status_code
            if 200 <= status < 300:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise RetrievalError("invalid", f"检索响应非法 JSON（HTTP {status}）: {exc}")
                if not isinstance(body, dict) or "result" not in body:
                    raise RetrievalError("invalid", f"检索响应缺少 result 字段（HTTP {status}）")
                return body
            if status in (401, 403):
                raise RetrievalError("auth", f"检索鉴权失败（HTTP {status}）")
            if status == 429:
                last_error = RetrievalError("rate_limit", f"检索限流（HTTP {status}）")
            elif status in _RETRYABLE_STATUS:
                last_error = RetrievalError("server", f"检索服务暂时不可用（HTTP {status}）")
            else:
                raise RetrievalError("invalid", f"检索返回非 2xx（HTTP {status}）")
        if attempt < max_retries:
            time.sleep(backoff * (2 ** attempt))
    raise last_error
