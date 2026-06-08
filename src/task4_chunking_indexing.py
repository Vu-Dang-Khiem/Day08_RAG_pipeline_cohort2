"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store

Chunking: dùng langchain-text-splitters (README khuyến khích).
Embedding: dùng sentence-transformers (README gợi ý).
Vector store: ChromaDB (README liệt kê là alternative đơn giản, local).

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "drug_law_docs"

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Chunking strategy: MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
# (cả 2 đều từ langchain-text-splitters mà README khuyến khích).
# Lý do: văn bản pháp luật (Điều/Khoản) và bài báo đều có heading rõ ràng.
#   - Bước 1 tách theo header (#, ##, ###) → mỗi chunk giữ trọn ngữ nghĩa 1 mục
#   - Bước 2 RecursiveCharacterTextSplitter cắt nhỏ phần quá dài
CHUNK_SIZE = 800        # ~800 ký tự: đủ ngữ cảnh 1 điều luật, không quá loãng
CHUNK_OVERLAP = 100     # overlap 100 để không mất ngữ cảnh ở biên chunk
CHUNKING_METHOD = "MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter"

# Embedding model: sentence-transformers/all-MiniLM-L6-v2 (README gợi ý: "nhẹ, nhanh")
# Lý do: chạy local qua sentence-transformers, 384 dim, nhanh, không tốn API.
# Muốn chất lượng tiếng Việt cao hơn → đổi thành "BAAI/bge-m3" (1024 dim, multilingual),
# code không cần sửa gì khác ngoài 2 dòng dưới.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector Store: ChromaDB (local, không cần Docker). Weaviate là lựa chọn hybrid tốt hơn.
VECTOR_STORE = "chromadb"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"

        # Bỏ YAML frontmatter (--- ... ---) để không ảnh hưởng embedding
        if content.startswith("---"):
            end_front = content.find("---", 3)
            if end_front != -1:
                content = content[end_front + 3:].strip()

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            }
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng langchain-text-splitters:
        1. MarkdownHeaderTextSplitter — tách theo H1/H2/H3
        2. RecursiveCharacterTextSplitter — cắt tiếp phần quá dài

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        # Bước 1: tách theo markdown header
        try:
            header_docs = header_splitter.split_text(doc["content"])
        except Exception:
            # Nếu không có header → coi cả document là 1 section
            from langchain_core.documents import Document
            header_docs = [Document(page_content=doc["content"], metadata={})]

        # Bước 2: cắt nhỏ từng section
        for hdoc in header_docs:
            sub_texts = char_splitter.split_text(hdoc.page_content)
            for i, chunk_text in enumerate(sub_texts):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        **hdoc.metadata,   # h1/h2/h3 headers
                        "chunk_index": i,
                    }
                })

    return chunks


def _get_embedding_function():
    """Tạo ChromaDB embedding function dùng sentence-transformers (model đã chọn)."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )


def embed_and_index(chunks: list[dict]):
    """
    Embed chunks bằng sentence-transformers model và lưu vào ChromaDB.
    ChromaDB quản lý embedding qua SentenceTransformerEmbeddingFunction.
    """
    import chromadb

    print(f"\nLoading embedding model: {EMBEDDING_MODEL} (sentence-transformers)")
    embedding_fn = _get_embedding_function()

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    # Xoá collection cũ để index lại từ đầu
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Đã xoá collection cũ: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 32
    total = len(chunks)
    print(f"Indexing {total} chunks vào ChromaDB (batch={batch_size})...")

    for start in range(0, total, batch_size):
        batch = chunks[start: start + batch_size]
        ids = [f"chunk_{start + j}" for j in range(len(batch))]
        documents_batch = [c["content"] for c in batch]
        metadatas_batch = [
            {k: str(v) for k, v in c["metadata"].items()} for c in batch
        ]
        collection.add(ids=ids, documents=documents_batch, metadatas=metadatas_batch)
        done = min(start + batch_size, total)
        print(f"  [{done}/{total}] indexed", end="\r")

    print(f"\n✓ Indexed {collection.count()} chunks vào collection '{COLLECTION_NAME}'")
    return collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD}")
    print(f"           : size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  Embedding: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dim)")
    print(f"  Store    : {VECTOR_STORE} → {CHROMA_PERSIST_DIR}")
    print("=" * 60)

    docs = load_documents()
    if not docs:
        print("✗ Không có file trong data/standardized/. Chạy Task 3 trước.")
        return False
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")
    avg_len = sum(len(c["content"]) for c in chunks) / max(len(chunks), 1)
    print(f"  Avg chunk length: {avg_len:.0f} chars")

    collection = embed_and_index(chunks)
    print(f"\n✓ Pipeline hoàn tất! {collection.count()} vectors trong ChromaDB.")

    # Smoke test
    print("\n--- Smoke test (query: 'tội phạm ma tuý hình phạt') ---")
    results = collection.query(query_texts=["tội phạm ma tuý hình phạt"], n_results=3)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"  [{i+1}] {meta.get('source', '?')} | {doc[:90]}...")

    return True


if __name__ == "__main__":
    run_pipeline()
