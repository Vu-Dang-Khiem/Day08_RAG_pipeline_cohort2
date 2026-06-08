"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search
    2. Merge kết quả bằng RRF (Reciprocal Rank Fusion)
    3. Rerank (cross-encoder)
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import sys

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

        Query
          ├→ Semantic Search → dense_results
          ├→ Lexical Search  → sparse_results
          ├→ Merge (RRF)     → merged
          ├→ Rerank          → final
          └→ if best_score < threshold → PageIndex fallback

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có rerank hay không

    Returns:
        List of {'content', 'score', 'metadata', 'source'} với source ∈ {'hybrid','pageindex'}
    """
    # Step 1: semantic + lexical (lấy rộng hơn để merge/rerank)
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: rerank (gán lại score thực để so threshold)
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Step 4: kiểm tra threshold → fallback PageIndex
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid best score ({best_score:.3f}) < threshold "
            f"({score_threshold}) → thử fallback PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]
        # PageIndex chưa cấu hình → trả hybrid hiện có (đỡ rỗng)
        return final_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            src = r.get("source", "?")
            meta_src = r.get("metadata", {}).get("source", "?")
            print(f"  {i}. [{r['score']:.3f}] [{src}] {meta_src}")
            print(f"     {r['content'][:90]}...")
