import re

def tokenize(text: str) -> list[str]:
    """去标点、转小写的标准英文/数字分词（改进版）"""
    return re.findall(r"[a-z0-9]+", text.lower())

def whitespace_tokenize(text: str) -> list[str]:
    """与 tau2-bench 内置 BM25 一致的简单空格分词（基线对比）"""
    return text.lower().split()
