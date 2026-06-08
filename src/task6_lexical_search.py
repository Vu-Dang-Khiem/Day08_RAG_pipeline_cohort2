"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25 (rank-bm25).

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Corpus được load từ ChromaDB collection (cùng nguồn với Task 5) để đảm bảo
semantic search và lexical search chạy trên cùng tập chunks.
"""

import re
from pathlib import Path

CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "drug_law_docs"

# Cache
CORPUS: list[dict] = []
_bm25 = None


def _tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản cho tiếng Việt: lowercase + tách theo non-word.
    (Có thể nâng cấp bằng underthesea/pyvi để tách từ ghép chính xác hơn.)
    """
    text = text.lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return tokens


def _load_corpus() -> list[dict]:
    """Load toàn bộ chunks từ ChromaDB collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    data = collection.get(include=["documents", "metadatas"])
    corpus = []
    for content, meta in zip(data["documents"], data["metadatas"]):
        corpus.append({"content": content, "metadata": meta or {}})
    return corpus


def build_bm25_index(corpus: list[dict] | None = None):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}.
                Nếu None → tự load từ ChromaDB.
    """
    global CORPUS, _bm25
    from rank_bm25 import BM25Okapi

    if corpus is None:
        corpus = _load_corpus()

    CORPUS = corpus
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by BM25 score descending.
    """
    global _bm25
    if _bm25 is None:
        try:
            build_bm25_index()
        except Exception as e:
            print(f"⚠ Không xây được BM25 index (chạy Task 4 trước?): {e}")
            return []

    if not CORPUS:
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)

    # Lấy top_k indices theo score giảm dần
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in indexed[:top_k]:
        if score <= 0:
            continue
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(score),
            "metadata": CORPUS[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    test_query = "Điều tàng trữ trái phép chất ma tuý hình phạt"
    print(f"Query: {test_query}\n")
    results = lexical_search(test_query, top_k=5)
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "?")
        print(f"[{i}] score={r['score']:.3f} | {src}")
        print(f"    {r['content'][:120]}...\n")
