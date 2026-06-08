"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://vietnamnet.vn/3-nu-nghe-si-viet-tu-huy-danh-tieng-vi-lien-quan-den-ma-tuy-2514737.html",
    "https://tienphong.vn/nhieu-nghe-si-viet-bi-bat-vi-dinh-vao-ma-tuy-post1649760.tpo",
    "https://baochinhphu.vn/khoi-to-bat-tam-giam-ca-si-long-nhat-son-ngoc-minh-vi-to-chuc-su-dung-ma-tuy-102260520125739676.htm",
    "https://nld.com.vn/showbiz-viet-nhung-nghe-si-gay-soc-vi-be-boi-ma-tuy-196250725113547841.htm",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def make_filename(index: int, url: str) -> str:
    """Tạo tên file từ index và URL."""
    domain = re.sub(r"https?://(www\.)?", "", url).split("/")[0]
    domain = re.sub(r"[^a-z0-9]", "-", domain)
    return f"article_{index:02d}_{domain}.json"


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)

        title = "Unknown"
        if result.metadata:
            title = result.metadata.get("title", "Unknown")

        # Fallback: extract title from markdown H1
        if title == "Unknown" and result.markdown:
            lines = result.markdown.strip().splitlines()
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    print("\n=== Task 2: Crawl bài báo ===\n")
    success_count = 0

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] {url}")

        filepath = DATA_DIR / make_filename(i, url)
        if filepath.exists():
            print(f"  ⏭ Đã tồn tại, bỏ qua: {filepath.name}")
            success_count += 1
            continue

        try:
            article = await crawl_article(url)

            if not article["content_markdown"]:
                print(f"  ✗ Không lấy được nội dung")
                continue

            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            content_len = len(article["content_markdown"])
            print(f"  ✓ Saved: {filepath.name} | Title: {article['title'][:60]} | {content_len} chars")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Lỗi: {e}")

        await asyncio.sleep(2)

    print(f"\n=== Kết quả: {success_count}/{len(ARTICLE_URLS)} bài báo đã crawl ===")

    existing = list(DATA_DIR.glob("*.json"))
    print(f"Files trong {DATA_DIR}:")
    for f in existing:
        print(f"  • {f.name}")

    return success_count >= 5


if __name__ == "__main__":
    asyncio.run(crawl_all())
