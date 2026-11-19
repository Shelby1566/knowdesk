"""把检索命中格式化成模型该看见的章节，而不是整篇政策。"""

from kb.chunk import Chunk


def format_matched_section(chunk: Chunk) -> str:
    """H2 章节正文加面包屑。全文不进入这段文本。"""
    body = (chunk.body or "").strip()
    if not body:
        body = (chunk.text or "").strip()
    return (
        f"{chunk.breadcrumb}\n"
        f"{body}\n\n"
        f"[matched_section doc_id={chunk.doc_id} chunk_id={chunk.chunk_id}; "
        f"this is the matched clause, not the full policy document]"
    )
