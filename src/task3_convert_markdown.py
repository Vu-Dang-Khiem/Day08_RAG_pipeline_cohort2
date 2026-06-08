"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, HTML, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import sys
from pathlib import Path

from markitdown import MarkItDown

sys.stdout.reconfigure(encoding="utf-8")

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# MarkItDown hỗ trợ: PDF, DOCX, PPTX, HTML, CSV, JSON, ...
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".pptx"}


def convert_legal_docs():
    """Convert PDF/DOCX/HTML files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    count = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.name == ".gitkeep":
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        if output_path.exists():
            print(f"  ⏭ Đã tồn tại: {output_path.name}")
            count += 1
            continue

        print(f"  → Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
            text = result.text_content.strip()

            if not text:
                print(f"    ⚠ Nội dung rỗng, bỏ qua")
                continue

            # Thêm metadata header
            header = f"---\nsource: {filepath.name}\ntype: legal\n---\n\n"
            output_path.write_text(header + text, encoding="utf-8")
            print(f"    ✓ Saved: {output_path.name} ({len(text):,} chars)")
            count += 1
        except Exception as e:
            print(f"    ✗ Lỗi: {e}")

    return count


def convert_news_articles():
    """
    Convert JSON crawled articles trong data/landing/news/ sang markdown.
    JSON đã chứa content_markdown từ Crawl4AI → chỉ cần extract và format.
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        if output_path.exists():
            print(f"  ⏭ Đã tồn tại: {output_path.name}")
            count += 1
            continue

        print(f"  → Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            title = data.get("title", "Bài báo không rõ tiêu đề")
            url = data.get("url", "")
            date_crawled = data.get("date_crawled", "")
            content = data.get("content_markdown", "").strip()

            if not content:
                print(f"    ⚠ Nội dung rỗng, bỏ qua")
                continue

            # Format markdown với metadata header
            header = (
                f"---\n"
                f"source: {filepath.name}\n"
                f"type: news\n"
                f"url: {url}\n"
                f"date_crawled: {date_crawled}\n"
                f"---\n\n"
                f"# {title}\n\n"
                f"**Nguồn:** {url}  \n"
                f"**Ngày crawl:** {date_crawled}\n\n"
                f"---\n\n"
            )

            output_path.write_text(header + content, encoding="utf-8")
            print(f"    ✓ Saved: {output_path.name} ({len(content):,} chars)")
            count += 1
        except Exception as e:
            print(f"    ✗ Lỗi: {e}")

    return count


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 55)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 55)

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    total = legal_count + news_count
    print(f"\n=== Kết quả: {total} files đã convert ({legal_count} legal, {news_count} news) ===")
    print(f"Output tại: {OUTPUT_DIR}")

    # Liệt kê files đã tạo
    print("\nFiles đã tạo:")
    for f in sorted(OUTPUT_DIR.rglob("*.md")):
        rel = f.relative_to(OUTPUT_DIR)
        print(f"  • {rel} ({f.stat().st_size / 1024:.0f} KB)")

    return total > 0


if __name__ == "__main__":
    convert_all()
