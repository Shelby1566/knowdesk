import re
import pytest
from kb.loader import Document, load_documents, DEFAULT_DOCS_DIR
from kb.chunk import Chunk, chunk_document, chunk_corpus

def test_single_doc_with_intro_and_two_h2():
    """Test case 1: 含引言 + 2个H2的文档，切出3块，面包屑正确"""
    doc = Document(
        doc_id="credit_card_01",
        title="Platinum Card Policy",
        content="Intro paragraph here.\n\n## Annual Fees\nFee is $95.\n\n## Foreign Transaction\nNo FX fee.",
        category="credit_card",
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 3
    assert chunks[0].breadcrumb == "Platinum Card Policy"
    assert chunks[0].body == "Intro paragraph here."
    assert chunks[1].breadcrumb == "Platinum Card Policy > Annual Fees"
    assert "Fee is $95." in chunks[1].text
    assert chunks[1].text.startswith("Platinum Card Policy > Annual Fees\n")
    assert chunks[2].breadcrumb == "Platinum Card Policy > Foreign Transaction"

def test_code_block_hash_not_split():
    """Edge case test: 代码块里的 ## 注释绝不能被当成 H2 切开"""
    doc = Document(
        doc_id="api_doc_01",
        title="API Guide",
        content="## Usage\nRun this:\n```bash\n## not a header\necho 1\n```\nDone.",
        category="api_doc",
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert "## not a header" in chunks[0].body

def test_full_corpus_698_docs_and_determinism():
    """验收 ②③④：全库698篇一篇不丢、正文拼接无损、两次运行确定性一致"""
    if not DEFAULT_DOCS_DIR.exists():
        pytest.skip(f"Corpus docs not found at {DEFAULT_DOCS_DIR}")
    docs = load_documents()
    assert len(docs) == 698

    chunks_run1 = chunk_corpus(docs)
    chunks_run2 = chunk_corpus(docs)

    # ④ 确定性
    assert [c.chunk_id for c in chunks_run1] == [c.chunk_id for c in chunks_run2]

    # ② 698篇文档的 doc_id 一篇不丢
    doc_ids_in_chunks = {c.doc_id for c in chunks_run1}
    assert len(doc_ids_in_chunks) == 698

    # ③ 块正文拼回后，所有非空正文的词不丢失
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks_run1:
        by_doc.setdefault(c.doc_id, []).append(c)

    for doc in docs:
        reconstructed = "\n".join(c.body for c in by_doc[doc.doc_id])
        # 原文去掉 ## 标题行后的每一行非空内容都应在 reconstructed 中
        in_code = False
        for line in doc.content.replace("\r\n", "\n").split("\n"):
            if line.strip().startswith("```"):
                in_code = not in_code
            if not in_code and re.match(r"^##\s+", line):
                continue
            if line.strip():
                assert line.strip() in reconstructed
