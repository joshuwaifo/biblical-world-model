"""
Tool registry — maps tool names to instances.

Tools are loaded lazily so startup doesn't fail if optional deps are missing.
The navigator queries this registry to discover available tools.

To add a new tool: import it and call register().
"""

from tools.base import Tool

_registry: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _registry[tool.name] = tool


def get(name: str) -> Tool:
    if name not in _registry:
        raise KeyError(f"Tool not registered: {name!r}. Available: {list(_registry)}")
    return _registry[name]


def all_tools() -> dict[str, Tool]:
    return dict(_registry)


def available_names() -> list[str]:
    return list(_registry.keys())


def _load_defaults() -> None:
    from tools.web_search import WebSearchTool
    from tools.fetch import FetchTool
    from tools.model import OllamaTool, OllamaVisionTool

    register(WebSearchTool())
    register(FetchTool())
    register(OllamaTool())

    # Vision tool registered only if a vision model is available
    try:
        vt = OllamaVisionTool()
        _ = vt.active_model   # raises if none available
        register(vt)
    except RuntimeError:
        pass   # no vision model pulled yet


# Load defaults on import
_load_defaults()
