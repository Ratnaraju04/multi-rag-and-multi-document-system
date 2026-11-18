"""
Web Scraper Loader for Multi-Modal RAG.
Extracts clean textual content, titles, metadata, tables, and linked images from web pages.
"""

import re
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

class WebScraperLoader:
    def __init__(self, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0.0.0 Safari/537.36"
            )
        }
        self.timeout = timeout

    def scrape_url(self, url: str):
        """
        Scrapes a URL and returns (documents, readable_source_label).
        documents: list of structured chunk dicts
        readable_source_label: a short, human-friendly name for the Document Focus dropdown
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            print(f"[WebScraperLoader] Error fetching {url}: {e}")
            return [], url

        soup = BeautifulSoup(html, "html.parser")

        # Remove irrelevant tags
        for element in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
            element.extract()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Build a readable, short source label for the Document Focus dropdown
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path_slug = parsed.path.rstrip("/").split("/")[-1].replace("_", " ").replace("-", " ").strip()
        if path_slug and len(path_slug) > 2:
            readable_source = f"{path_slug} ({domain})"
        else:
            readable_source = domain if domain else title[:80]

        # --- Extract infobox / structured tables FIRST (Wikipedia, IMDb, etc.) ---
        infobox_text_parts = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    key = cells[0].get_text(separator=" ", strip=True)
                    value = cells[1].get_text(separator=" ", strip=True)
                    if key and value and len(value) > 1:
                        infobox_text_parts.append(f"{key}: {value}")
                elif len(cells) == 1:
                    text = cells[0].get_text(separator=" ", strip=True)
                    if len(text) > 5:
                        infobox_text_parts.append(text)
        infobox_text = "\n".join(infobox_text_parts)

        # Extract main body text from paragraphs, headings, and list items
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "article", "li", "span", "td", "th"])]
        body_text = "\n\n".join([p for p in paragraphs if len(p) > 20])

        # Combine: infobox first (most fact-dense), then body text
        full_text = (infobox_text + "\n\n" + body_text).strip() if infobox_text else body_text

        # Extract images found on the page
        extracted_images = []
        for img in soup.find_all("img"):
            src = img.get("src")
            alt = img.get("alt", "").strip()
            if src:
                # Handle relative URLs
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    from urllib.parse import urljoin
                    src = urljoin(url, src)
                if src.startswith("http"):
                    extracted_images.append({"url": src, "caption": alt})

        # Break text into chunks
        chunks = self._chunk_text(full_text, chunk_size=800, overlap=100)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"web_{re.sub(r'[^a-zA-Z0-9]', '_', url)[:40]}_{i}",
                "modality": "text",
                "content": f"[Source: {title} ({url})]\n{chunk}",
                "metadata": {
                    "source": readable_source,
                    "url": url,
                    "title": title,
                    "chunk_index": i,
                    "type": "web_page",
                    "total_images_found": len(extracted_images)
                }
            })

        # Also register image candidates as image-reference chunks
        for j, img in enumerate(extracted_images[:5]):  # top 5 relevant images
            if img["caption"]:
                documents.append({
                    "id": f"web_img_{re.sub(r'[^a-zA-Z0-9]', '_', url)[:30]}_{j}",
                    "modality": "image_url",
                    "content": f"Image from {title}: {img['caption']} (URL: {img['url']})",
                    "image_url": img["url"],
                    "metadata": {
                        "source": readable_source,
                        "url": url,
                        "title": title,
                        "type": "web_image",
                        "image_url": img["url"],
                        "caption": img["caption"]
                    }
                })

        return documents, readable_source

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        words = text.split()
        if not words:
            return []
        
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks
