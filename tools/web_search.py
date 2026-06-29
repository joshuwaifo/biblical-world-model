"""
Web search tool — DuckDuckGo (no API key required).

Returns up to `max_results` snippets.  Each snippet is a ToolResult with
modality="text".  The navigator embeds each snippet into the Biblical
coordinate space and uses the geometry to decide what to search next.
"""

from tools.base import Tool, ToolResult


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for information. "
        "Returns page titles, URLs, and text snippets."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query":       {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    modalities_in  = ["text"]
    modalities_out = ["text"]

    def invoke(self, query: str, max_results: int = 5) -> list[ToolResult]:
        from duckduckgo_search import DDGS
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    body = r.get("body", "") or ""
                    title = r.get("title", "") or ""
                    url = r.get("href", "") or ""
                    text = f"{title}\n{body}".strip()
                    results.append(ToolResult(
                        content=text,
                        modality="text",
                        raw_bytes=text.encode("utf-8", errors="replace"),
                        source=url,
                        metadata={"title": title, "url": url, "query": query},
                    ))
        except Exception as e:
            results.append(ToolResult(
                content=f"Search failed: {e}",
                modality="text",
                raw_bytes=b"",
                source="duckduckgo",
                metadata={"error": str(e)},
            ))
        return results
