"""
Tool abstraction layer.

Any external capability (web search, URL fetch, foundation model, image
generator, code executor, sensor, database, API) is a Tool.  The navigator
selects and invokes tools based on guidance from the Biblical world model
geometry — the geometry tells it WHERE to look; the tools do the looking.

Adding a new tool:
  1. Subclass Tool and implement invoke().
  2. Register in tools/registry.py.
  3. The navigator discovers it automatically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    content: Any             # parsed output (str, dict, bytes, PIL.Image, ...)
    modality: str            # "text" | "image" | "audio" | "video" | "bytes" | "json"
    raw_bytes: bytes         # always preserve original bytes for re-encoding
    source: str              # human-readable provenance (URL, model name, etc.)
    metadata: dict = field(default_factory=dict)


class Tool(ABC):
    name: str                # machine key, e.g. "web_search"
    description: str         # human-readable, used by foundation model as context
    input_schema: dict       # JSON Schema for the kwargs accepted by invoke()
    modalities_in: list[str] # modalities this tool accepts as input
    modalities_out: list[str]# modalities this tool can produce

    @abstractmethod
    def invoke(self, **kwargs) -> ToolResult:
        """Execute the tool and return a ToolResult."""
