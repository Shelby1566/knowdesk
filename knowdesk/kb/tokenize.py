import re

def tokenize(text: str) -> list[str]:
    """去标点、转小写的标准英文/数字及 CJK 单字分词（支持中英文多语言）"""
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text.lower())

def whitespace_tokenize(text: str) -> list[str]:
    """与 tau2-bench 内置 BM25 一致的简单空格分词（基线对比）"""
    return text.lower().split()
