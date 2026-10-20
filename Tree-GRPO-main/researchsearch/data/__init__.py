"""统一数据层：多源合并、去重、全局唯一 index 与 manifest。

核心逻辑在 researchsearch.data.build，直接导入子模块即可：

    from researchsearch.data.build import build_dataset, build_sample, load_jsonl_source

命令行入口：python -m researchsearch.data.build --help
"""
