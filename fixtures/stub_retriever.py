#!/usr/bin/env python3
"""最小 stub 检索服务：POST /retrieve 返回固定 Wiki 格式。

供 Ray worker probe（只需 TCP 可达）与 GRPO/Tree-GRPO 更新闭环（需正确
``{"result": [[{"document": {"id", "contents"}, "score"}]]}`` 结构）使用，
不依赖真实语料/索引。
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# 关键词 → 文档内容的简单映射，让 smoke 中的简单事实问题能命中非空答案。
_KNOWLEDGE = [
    ("harry potter", "J.K. Rowling is the author of the Harry Potter series."),
    ("capital of france", "Paris is the capital of France."),
    ("windows", "Microsoft developed the Windows operating system."),
    ("largest planet", "Jupiter is the largest planet in the solar system."),
]


def _answer_for(query):
    lowered = query.lower()
    for key, contents in _KNOWLEDGE:
        if key in lowered:
            return contents
    return "No specific information is available for this query."


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        queries = body.get("queries", [])
        topk = int(body.get("topk", 3))
        result = []
        for query in queries:
            docs = [
                {
                    "document": {"id": "wiki:stub", "contents": _answer_for(query)},
                    "score": 1.0,
                }
                for _ in range(min(max(topk, 1), 3))
            ]
            result.append(docs)
        payload = json.dumps({"result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass  # 静默，避免刷日志


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
