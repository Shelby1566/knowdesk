import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DOCS_DIR = Path(__file__).resolve().parents[2] / "tau2-bench/data/tau2/domains/banking_knowledge/documents"
DEFAULT_TASKS_PATH = Path(__file__).resolve().parents[2] / "tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"

@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str
    category: str

def extract_category(doc_id: str) -> str:
    """从 doc_id 提取子类前缀（去掉末尾的 _数字 或纯数字序号）"""
    return re.sub(r"[_-]?\d+$", "", doc_id)

def load_documents(docs_dir: Path = DEFAULT_DOCS_DIR) -> list[Document]:
    docs: list[Document] = []
    for p in sorted(docs_dir.glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        doc_id = str(raw["id"])
        docs.append(
            Document(
                doc_id=doc_id,
                title=str(raw["title"]).strip(),
                content=str(raw["content"]),
                category=extract_category(doc_id),
            )
        )
    return docs
