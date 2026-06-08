"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "Tôi không thể xác minh thông tin này"
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: số chunks đưa vào context. Chọn 5 vì đủ evidence mà không quá dài
# (context dài → lost in the middle, tốn token).
TOP_K = 5

# top_p (nucleus sampling): Chọn 0.9 — đủ tự nhiên nhưng vẫn bám context,
# không quá ngẫu nhiên (RAG cần factual).
TOP_P = 0.9

# temperature: 0.3 — RAG cần factual, ít sáng tạo, hạn chế bịa.
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.
Trả lời câu hỏi một cách đầy đủ bằng tiếng Việt, CHỈ dựa trên context được cung cấp.

Với MỖI khẳng định/sự kiện, chèn ngay citation trong ngoặc vuông trỏ tới nguồn cụ thể,
ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [VnExpress, 2024].

Nếu thông tin KHÔNG có trong context, hãy trả lời 'Tôi không thể xác minh thông tin này
từ nguồn hiện có' thay vì đoán.

Quy tắc:
- Chỉ dùng thông tin trong context
- Mọi khẳng định PHẢI có citation
- Nếu context không đủ, nói rõ
- Trình bày thành đoạn rõ ràng"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle".

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI, quên ở GIỮA.
    Strategy: quan trọng nhất ở đầu, nhì ở cuối, kém quan trọng dồn vào giữa.

    Input (theo score):  [1, 2, 3, 4, 5]
    Output:              [1, 3, 5, 4, 2]
    (chunks[0::2] giữ thứ tự + chunks[1::2] đảo ngược)

    Args:
        chunks: List sorted by score descending (từ retrieval)

    Returns:
        List reordered.
    """
    if len(chunks) <= 2:
        return list(chunks)

    first_half = chunks[0::2]          # vị trí chẵn: 1,3,5 → ra đầu
    second_half = chunks[1::2][::-1]   # vị trí lẻ đảo ngược: 4,2 → ra cuối
    return first_half + second_half


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt, có label source để LLM cite.

    Args:
        chunks: List of {'content', 'metadata', 'score'}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# LLM CALL
# =============================================================================

def _real_key(val: str) -> bool:
    v = (val or "").strip()
    return bool(v) and "xxx" not in v.lower() and v not in ("your_api_key",)


def _call_llm(system_prompt: str, user_message: str) -> str | None:
    """
    Gọi LLM. Ưu tiên OpenAI (theo README), sau đó Anthropic Claude.
    Trả None nếu không có API key thật.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if _real_key(openai_key):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠ OpenAI lỗi ({type(e).__name__}: {e}). Thử Claude / fallback.")

    if _real_key(anthropic_key):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return resp.content[0].text
        except Exception as e:
            print(f"  ⚠ Anthropic lỗi ({type(e).__name__}: {e}). Dùng fallback.")

    return None


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks (Task 9)
        2. Reorder tránh lost in the middle
        3. Format context với source labels
        4. Build prompt + call LLM
        5. Return answer + sources

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str  # 'hybrid' | 'pageindex' | 'none'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt + call LLM
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"
    answer = _call_llm(SYSTEM_PROMPT, user_message)

    if answer is None:
        # Không có API key → trả lời extractive từ context (vẫn có citation nguồn)
        top = chunks[0]
        src = top.get("metadata", {}).get("source", "nguồn")
        answer = (
            "⚠ Chưa cấu hình API key LLM (OPENAI_API_KEY/ANTHROPIC_API_KEY) nên "
            "chưa thể sinh câu trả lời tự nhiên. Dưới đây là đoạn liên quan nhất:\n\n"
            f"{top['content'][:500]} [{src}]"
        )

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer'][:600]}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
