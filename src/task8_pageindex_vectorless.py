"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng structural
understanding (cây mục lục) của document thay vì embedding.

Cài đặt:
    pip install pageindex

Luồng dùng (theo PageIndexClient SDK 0.2.x):
    client = PageIndexClient(api_key)
    res = client.submit_document(file_path)        # → doc_id
    client.is_retrieval_ready(doc_id)              # chờ index xong
    q = client.submit_query(doc_id, query)         # → retrieval_id
    client.get_retrieval(retrieval_id)             # → kết quả

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai, lấy API key → đặt PAGEINDEX_API_KEY trong .env
    2. Upload documents (chạy upload_documents())
    3. Query qua pageindex_search()
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
# PageIndex chỉ hỗ trợ PDF — upload file gốc từ landing/legal
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
# Lưu mapping doc_id ↔ filename sau khi upload để query lại
DOC_REGISTRY = Path(__file__).parent.parent / "data" / "pageindex_docs.json"


def _key_configured() -> bool:
    """True nếu API key thật (không rỗng, không phải placeholder)."""
    k = PAGEINDEX_API_KEY.strip()
    return bool(k) and "xxx" not in k.lower() and k not in ("pi_xxx", "your_api_key")


def _get_client():
    """Tạo PageIndexClient. None nếu chưa cài SDK hoặc chưa có API key thật."""
    if not _key_configured():
        return None
    try:
        from pageindex import PageIndexClient
        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception as e:
        print(f"⚠ Không khởi tạo được PageIndexClient: {e}")
        return None


def upload_documents() -> dict:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    Lưu mapping {doc_id: filename} vào DOC_REGISTRY.

    Returns:
        dict {doc_id: filename}
    """
    client = _get_client()
    if client is None:
        print("⚠ Chưa có PAGEINDEX_API_KEY hoặc SDK. Bỏ qua upload.")
        return {}

    registry: dict[str, str] = {}
    # PageIndex chỉ hỗ trợ PDF — upload file PDF gốc
    pdf_files = sorted(LEGAL_DIR.glob("*.pdf"))
    if not pdf_files:
        print("⚠ Không tìm thấy PDF trong data/landing/legal/")
        return {}

    for pdf_file in pdf_files:
        try:
            print(f"  → Uploading: {pdf_file.name}")
            res = client.submit_document(file_path=str(pdf_file))
            doc_id = res.get("doc_id") or res.get("id") or res.get("document_id")
            if doc_id:
                registry[doc_id] = pdf_file.name
                print(f"    ✓ doc_id={doc_id}")
            else:
                print(f"    ⚠ Không lấy được doc_id từ response: {res}")
        except Exception as e:
            print(f"    ✗ Lỗi upload {pdf_file.name}: {e}")

    DOC_REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Đã upload {len(registry)} docs. Registry: {DOC_REGISTRY}")
    return registry


def _load_registry() -> dict:
    if DOC_REGISTRY.exists():
        return json.loads(DOC_REGISTRY.read_text(encoding="utf-8"))
    return {}


def _extract_nodes(retrieval_result: dict, filename: str) -> list[dict]:
    """Parse kết quả get_retrieval thành list chuẩn (defensive với nhiều shape)."""
    items = []
    # Các shape khả dĩ: {'retrieval':[...]}, {'results':[...]}, {'nodes':[...]}, {'sources':[...]}
    candidates = (
        retrieval_result.get("retrieval")
        or retrieval_result.get("results")
        or retrieval_result.get("nodes")
        or retrieval_result.get("sources")
        or []
    )
    for c in candidates:
        if isinstance(c, dict):
            content = (
                c.get("text") or c.get("content") or c.get("node_text")
                or c.get("relevant_content") or ""
            )
            score = c.get("score") or c.get("relevance_score") or 0.0
        else:
            content, score = str(c), 0.0
        if content:
            items.append({
                "content": content,
                "score": float(score) if score else 0.0,
                "metadata": {"source": filename},
                "source": "pageindex",
            })
    return items


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
        Trả [] nếu chưa cấu hình PageIndex (không crash pipeline).
    """
    client = _get_client()
    if client is None:
        return []

    registry = _load_registry()
    if not registry:
        # Chưa upload → thử upload trước
        registry = upload_documents()
        if not registry:
            return []

    all_results: list[dict] = []
    for doc_id, filename in registry.items():
        try:
            if not client.is_retrieval_ready(doc_id):
                # Chờ index (tối đa ~30s)
                for _ in range(15):
                    time.sleep(2)
                    if client.is_retrieval_ready(doc_id):
                        break

            q = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = q.get("retrieval_id") or q.get("id")
            if retrieval_id:
                result = client.get_retrieval(retrieval_id)
            else:
                result = q  # một số phiên bản trả kết quả trực tiếp
            all_results.extend(_extract_nodes(result, filename))
        except Exception as e:
            print(f"  ⚠ PageIndex query lỗi ({filename}): {e}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()
        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
