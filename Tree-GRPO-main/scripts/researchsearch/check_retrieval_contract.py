#!/usr/bin/env python3
"""真实 Wiki /retrieve 契约测试（第 1 周 T03 验收，第 2 周 §3.2 T03 转 ✅）。

对 RETRIEVER_URL 发真实 POST 请求，逐项校验检索契约（方案 §4.3 / Day 2 验收）：
- 单条查询返回形状正确（status 200，body 含 result）；
- 批量查询外层顺序与输入一一对应（返回组数 == 查询数）；
- 返回的 document.id 非空、可回查正文（document.contents 非空）；
- 中/英文字符查询均可用；
- 空结果返回合法空列表，不崩溃；
- 非法输入（缺 queries）返回 4xx 而非 500 崩溃。

用法（retriever env，服务已起）：
    RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
      python scripts/researchsearch/check_retrieval_contract.py

任意一项失败退出码非 0，便于开机自检 / CI。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _post(url: str, payload: dict, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except urllib.error.URLError as e:
        return None, f"URLError: {e.reason}"


def _iter_docs(body):
    # Wiki 返回形状：{"result":[[{"document":{"id","contents"},"score"}]]}
    for group in body.get("result", []):
        for item in group:
            yield item.get("document", {}), item.get("score")


def check_single_query(url):
    payload = {"queries": ["Who directed the film Jaws?"], "topk": 3, "return_scores": True}
    status, body = _post(url, payload)
    ok = status == 200 and isinstance(body, dict) and isinstance(body.get("result"), list)
    return ok, f"status={status}, result_is_list={isinstance(body, dict) and isinstance(body.get('result'), list)}"


def check_batch_order(url):
    queries = ["Who directed the film Jaws?", "What is the Eiffel Tower?"]
    payload = {"queries": queries, "topk": 3, "return_scores": True}
    status, body = _post(url, payload)
    if status != 200 or not isinstance(body, dict):
        return False, f"status={status}"
    result = body.get("result")
    ok = isinstance(result, list) and len(result) == len(queries)
    return ok, f"输入 {len(queries)} 条，返回组数 {len(result) if isinstance(result, list) else type(result).__name__}"


def check_doc_id_and_contents(url):
    payload = {"queries": ["Who directed the film Jaws?"], "topk": 3, "return_scores": True}
    status, body = _post(url, payload)
    if status != 200 or not isinstance(body, dict):
        return False, f"status={status}"
    docs = list(_iter_docs(body))
    if not docs:
        return False, "未返回任何文档"
    bad = [
        i for i, (doc, _) in enumerate(docs)
        if not doc.get("id") or not doc.get("contents")
    ]
    return (not bad), f"返回 {len(docs)} 条，缺 id/contents 的条数 {len(bad)}"


def check_chinese_query(url):
    payload = {"queries": ["埃菲尔铁塔位于哪个城市？"], "topk": 3, "return_scores": True}
    status, body = _post(url, payload)
    ok = status == 200 and isinstance(body, dict) and isinstance(body.get("result"), list)
    return ok, f"status={status}"


def check_empty_result(url):
    # 空查询列表应返回合法空结果、不崩溃。flat index 对非空查询总返回 top-k 最近邻
    # （即便不相关），真正的「空结果」场景是空输入列表。
    payload = {"queries": [], "topk": 3, "return_scores": True}
    status, body = _post(url, payload)
    if status != 200 or not isinstance(body, dict):
        return False, f"status={status}"
    result = body.get("result")
    ok = isinstance(result, list) and len(result) == 0
    return ok, f"空查询列表返回组数 {len(result) if isinstance(result, list) else type(result).__name__}（合法空列表）"


def check_invalid_input(url):
    # 缺 queries 字段：服务应返回 4xx（如 422），不应 500 崩溃。
    payload = {"topk": 3}
    status, _ = _post(url, payload)
    ok = status is not None and 400 <= status < 500
    return ok, f"缺 queries 返回 status={status}（期望 4xx 非 500）"


def main() -> int:
    ap = argparse.ArgumentParser(description="真实 Wiki /retrieve 契约测试")
    ap.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    args = ap.parse_args()

    checks = [
        ("单条查询形状", check_single_query),
        ("批量顺序对应", check_batch_order),
        ("doc_id 可回查正文", check_doc_id_and_contents),
        ("中文字符查询", check_chinese_query),
        ("空结果合法", check_empty_result),
        ("非法输入非 500", check_invalid_input),
    ]

    failed = 0
    for name, fn in checks:
        try:
            ok, detail = fn(args.url)
        except Exception as e:  # noqa: BLE001 —— 契约测试任何异常都记为失败
            ok, detail = False, f"异常: {type(e).__name__}: {e}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed += 0 if ok else 1

    print(f"\n契约测试：{len(checks) - failed}/{len(checks)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
