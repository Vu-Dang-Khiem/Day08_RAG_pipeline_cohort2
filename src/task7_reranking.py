"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp (README "Lựa chọn (chọn 1)"):
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

File này implement cả 3 để pipeline linh hoạt:
    - rerank_cross_encoder: Jina API (theo code mẫu README) → fallback local CrossEncoder
    - rerank_mmr: tự implement công thức MMR
    - rerank_rrf: tự implement công thức RRF

Phương pháp mặc định cho rerank(): cross_encoder.
"""

import math
import os
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# Cross-encoder local mặc định: BAAI/bge-reranker-base (~110MB, multilingual, tốt tiếng Việt)
# Ưu tiên: Jina API (nếu có key) → bge-reranker-base (local) → embedding-cosine fallback
_LOCAL_CE_MODEL = os.getenv("LOCAL_RERANKER_MODEL", "BAAI/bge-reranker-base")
_local_ce = None


# =============================================================================
# Cross-encoder reranking
# =============================================================================

def _jina_key_ok() -> bool:
    k = JINA_API_KEY.strip()
    return bool(k) and "xxx" not in k.lower() and k != "jina_xxx"


def _jina_api_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict] | None:
    """Gọi Jina Reranker API (theo đúng code mẫu README). None nếu không có key/lỗi."""
    if not _jina_key_ok():
        return None
    import requests

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        reranked = response.json()["results"]
        return [
            {**candidates[r["index"]], "score": float(r["relevance_score"])}
            for r in reranked
        ]
    except Exception as e:
        print(f"  ⚠ Jina API lỗi ({e}), chuyển sang local reranker")
        return None


def _local_cross_encoder_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict] | None:
    """Rerank bằng local CrossEncoder (BAAI/bge-reranker-base). None nếu không load được."""
    global _local_ce
    try:
        if _local_ce is None:
            from sentence_transformers import CrossEncoder
            print(f"  → Nạp local reranker: {_LOCAL_CE_MODEL} (lần đầu mất ~30s)...")
            _local_ce = CrossEncoder(_LOCAL_CE_MODEL, max_length=512)

        pairs = [[query, c["content"]] for c in candidates]
        scores = _local_ce.predict(pairs)

        scored = []
        for c, s in zip(candidates, scores):
            item = {**c, "score": float(s)}
            scored.append(item)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    except Exception as e:
        print(f"  ⚠ Local CrossEncoder lỗi ({e}), dùng fallback embedding-cosine")
        return None


def _embedding_cosine_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Fallback reranker: embed query + candidates bằng ChromaDB DefaultEmbeddingFunction
    (ONNX, không cần model nặng), tính cosine similarity, re-sort.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()
    texts = [query] + [c["content"] for c in candidates]
    embs = ef(texts)
    q_emb = embs[0]
    doc_embs = embs[1:]

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    scored = []
    for c, d_emb in zip(candidates, doc_embs):
        scored.append({**c, "score": cosine(q_emb, d_emb)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder.
    Thứ tự ưu tiên: Jina API → local CrossEncoder → embedding-cosine fallback.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by score descending.
    """
    if not candidates:
        return []

    result = _jina_api_rerank(query, candidates, top_k)
    if result is not None:
        return result

    result = _local_cross_encoder_rerank(query, candidates, top_k)
    if result is not None:
        return result

    return _embedding_cosine_rerank(query, candidates, top_k)


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content', 'score', 'embedding', 'metadata'}
        top_k: Số lượng kết quả
        lambda_param: Trade-off relevance (1.0) vs diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            emb = candidates[idx]["embedding"]
            relevance = _cosine_sim(query_embedding, emb)

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(emb, candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for rank, i in enumerate(selected):
        item = {**candidates[i]}
        item["score"] = float(len(selected) - rank)  # score giảm dần theo thứ tự chọn
        results.append(item)
    return results


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List các ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối
        k: Smoothing constant (default=60, Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = {**content_map[content]}
        item["score"] = float(score)
        results.append(item)
    return results


# =============================================================================
# Unified rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "cross_encoder" (mặc định)

    Returns:
        List of top_k reranked candidates.

    Lưu ý: MMR cần query_embedding, RRF cần nhiều ranked lists → gọi
    rerank_mmr() / rerank_rrf() trực tiếp khi cần.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise ValueError("MMR cần query_embedding — gọi rerank_mmr() trực tiếp")
    elif method == "rrf":
        raise ValueError("RRF cần nhiều ranked lists — gọi rerank_rrf() trực tiếp")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ ma tuý", "score": 0.6, "metadata": {}},
        {"content": "Hướng dẫn lập trình Python cơ bản", "score": 0.5, "metadata": {}},
    ]
    print("=== Cross-encoder rerank ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.3f} | {r['content']}")

    print("\n=== RRF (gộp 2 ranked lists) ===")
    list_a = dummy_candidates
    list_b = list(reversed(dummy_candidates))
    rrf_results = rerank_rrf([list_a, list_b], top_k=3)
    for i, r in enumerate(rrf_results, 1):
        print(f"[{i}] rrf={r['score']:.4f} | {r['content']}")
