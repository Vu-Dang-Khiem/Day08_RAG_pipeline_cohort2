"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Implementation: query ChromaDB collection đã index ở Task 4.
ChromaDB trả về distance (cosine) → đổi sang similarity score = 1 - distance.
"""

from pathlib import Path

CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "drug_law_docs"

# Dùng đúng embedding model đã index ở Task 4 (sentence-transformers)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache client + collection để không khởi tạo lại mỗi lần gọi
_collection = None


def _get_collection():
    """Lấy ChromaDB collection (cache lại sau lần đầu)."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    _collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL, device="cpu"
        ),
    )
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (0..1)
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    try:
        collection = _get_collection()
    except Exception as e:
        print(f"⚠ Không kết nối được ChromaDB (chạy Task 4 trước?): {e}")
        return []

    n = min(top_k, max(collection.count(), 1))
    results = collection.query(
        query_texts=[query],
        n_results=n,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for content, meta, dist in zip(documents, metadatas, distances):
        # cosine distance → similarity. Clamp về [0, 1].
        score = max(0.0, min(1.0, 1.0 - dist))
        output.append({
            "content": content,
            "score": float(score),
            "metadata": meta or {},
        })

    # Đảm bảo sorted descending
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    test_query = "hình phạt cho tội tàng trữ ma tuý"
    print(f"Query: {test_query}\n")
    results = semantic_search(test_query, top_k=5)
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "?")
        print(f"[{i}] score={r['score']:.3f} | {src}")
        print(f"    {r['content'][:120]}...\n")
