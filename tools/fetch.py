"""
URL fetch + parse tool.

Fetches a URL, returns raw bytes (for re-encoding into Biblical space) and
a best-effort text extraction.  The modality is auto-detected from the
Content-Type header: text/html → text, image/* → image, etc.
"""

import re
from tools.base import Tool, ToolResult


class FetchTool(Tool):
    name = "fetch"
    description = (
        "Fetch a URL and return its content. "
        "Auto-detects modality from Content-Type."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url":     {"type": "string"},
            "timeout": {"type": "number", "default": 10},
        },
        "required": ["url"],
    }
    modalities_in  = ["text"]   # input is always a URL string
    modalities_out = ["text", "image", "audio", "video", "bytes"]

    def invoke(self, url: str, timeout: float = 10) -> ToolResult:
        import httpx
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                             headers={"User-Agent": "WorldModelBot/0.1"})
            raw = resp.content
            ct = resp.headers.get("content-type", "")

            if "text" in ct:
                text = _strip_html(raw.decode("utf-8", errors="replace"))
                return ToolResult(content=text, modality="text",
                                  raw_bytes=raw, source=url,
                                  metadata={"content_type": ct, "status": resp.status_code})
            elif "image" in ct:
                return ToolResult(content=raw, modality="image",
                                  raw_bytes=raw, source=url,
                                  metadata={"content_type": ct})
            elif "audio" in ct:
                return ToolResult(content=raw, modality="audio",
                                  raw_bytes=raw, source=url,
                                  metadata={"content_type": ct})
            elif "video" in ct:
                return ToolResult(content=raw, modality="video",
                                  raw_bytes=raw, source=url,
                                  metadata={"content_type": ct})
            else:
                return ToolResult(content=raw, modality="bytes",
                                  raw_bytes=raw, source=url,
                                  metadata={"content_type": ct})
        except Exception as e:
            return ToolResult(content=f"Fetch failed: {e}", modality="text",
                              raw_bytes=b"", source=url,
                              metadata={"error": str(e)})


def _strip_html(html: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:8000]   # cap at 8k chars for encoding
