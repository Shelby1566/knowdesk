
from dataclasses import dataclass
from knowdesk.kb.loader import Document



@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    breadcrumb: str
    body: str
    text: str
    category: str

def chunk_document(doc: Document) -> list[Chunk]:
    content=doc.content.replace('\r\n', '\n')
    in_code_block = False
    chunks = []
    idx=0
    current_breadcrumb = doc.title
    buffer=[]
    for line in content.splitlines():
        if line.strip().startswith('```'):
            in_code_block = not in_code_block


        if line.startswith("## ") and  not in_code_block:
            body='\n'.join(buffer).strip()

            if body:
                chunk_id=f"{doc.doc_id}#{idx}"
                idx+=1
                chunk=Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    breadcrumb=current_breadcrumb,
                    body=body,
                    text=f"{current_breadcrumb}\n{body}",
                    category=doc.category
                )
                chunks.append(chunk)
            h2_title=line[3:].strip()
            current_breadcrumb=f"{doc.title} > {h2_title}"
          
            buffer=[]
            continue
        buffer.append(line)
    body='\n'.join(buffer).strip()
    if body:
        chunk_id=f"{doc.doc_id}#{idx}"
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            breadcrumb=current_breadcrumb,
            body=body,
            text=f"{current_breadcrumb}\n{body}",
            category=doc.category
        )
        chunks.append(chunk)
    if not chunks:
        chunk_id=f"{doc.doc_id}#0"
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            breadcrumb=doc.title,
            body="",
            text=doc.title,
            category=doc.category
        )
        chunks.append(chunk)
    return chunks


def chunk_corpus(docs: list[Document]) -> list[Chunk]:
    all_chunks = []

    for doc in docs:
        result = chunk_document(doc)
        all_chunks.extend(result)

    return all_chunks