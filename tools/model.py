"""
Foundation/language/omni model tool — Ollama-first, open-source by default.

Any model accessible via Ollama (Gemma, Llama, Mistral, EXAONE, Phi, Qwen,
etc.) is wrapped here.  Vision-capable models (LLaVA, moondream,
llama3.2-vision) accept image bytes and return text.

The navigator injects Biblical coordinate context into the system prompt so
the model is grounded in where it is in the information space before it
searches or reasons.

Adding a new model:
  1. `ollama pull <model-name>`
  2. Pass model_name to OllamaTool or register as a named preset below.
"""

import base64
from tools.base import Tool, ToolResult

# Default model preference order — first available is used
DEFAULT_MODEL_PREFERENCE = [
    "gemma3:4b",
    "gemma3:9b",
    "gemma3:2b",
    "gemma3:1b",
    "llama3.2:3b",
    "llama3.2:1b",
    "exaone-deep:32b",
    "mistral",
    "phi3",
]

DEFAULT_VISION_PREFERENCE = [
    "llama3.2-vision",
    "llava:13b",
    "llava:7b",
    "llava",
    "moondream",
    "bakllava",
]


def _available_models() -> list[str]:
    """Return names of models currently pulled in Ollama."""
    try:
        import ollama
        return [m.model for m in ollama.list().models]
    except Exception:
        return []


def resolve_model(preference: list[str]) -> str | None:
    """Return the first available model from a preference list."""
    available = set(_available_models())
    for m in preference:
        if m in available:
            return m
    # fallback: return whatever is available
    if available:
        return next(iter(available))
    return None


class OllamaTool(Tool):
    """
    Generic Ollama tool — wraps any locally-running open-source model.

    The model is selected by name (e.g. "gemma3:4b", "llama3.2:1b",
    "exaone-deep:32b").  If model_name is None, the first available model
    from DEFAULT_MODEL_PREFERENCE is used.
    """

    name = "ollama"
    description = (
        "Open-source language model running locally via Ollama. "
        "Accepts text prompt + optional context; returns text."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "prompt":     {"type": "string"},
            "context":    {"type": "string", "default": ""},
            "model_name": {"type": "string", "default": None},
        },
        "required": ["prompt"],
    }
    modalities_in  = ["text", "image", "bytes"]
    modalities_out = ["text"]

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name

    @property
    def active_model(self) -> str:
        if self._model_name:
            return self._model_name
        m = resolve_model(DEFAULT_MODEL_PREFERENCE)
        if m is None:
            raise RuntimeError(
                "No Ollama models available. "
                "Run: ollama pull gemma3:4b"
            )
        return m

    def invoke(
        self,
        prompt: str,
        *,
        context: str = "",
        image_bytes: bytes | None = None,
        audio_bytes: bytes | None = None,
        available_tools: list[str] | None = None,
        model_name: str | None = None,
        stream: bool = False,
    ) -> ToolResult:
        import ollama

        model = model_name or self.active_model
        messages: list[dict] = []

        if context:
            messages.append({"role": "system", "content": context})

        user_msg: dict = {"role": "user", "content": prompt}

        # Vision models accept base64-encoded images in the images field
        if image_bytes:
            user_msg["images"] = [image_bytes]   # ollama client accepts raw bytes

        messages.append(user_msg)

        try:
            resp = ollama.chat(model=model, messages=messages, stream=False)
            text = resp.message.content
        except Exception as e:
            text = f"[OllamaTool error — model={model}]: {e}"

        return ToolResult(
            content=text,
            modality="text",
            raw_bytes=text.encode("utf-8", errors="replace"),
            source=f"ollama/{model}",
            metadata={"model": model, "vision": image_bytes is not None},
        )


class OllamaVisionTool(OllamaTool):
    """
    Ollama tool specialised for vision/omni models.

    Accepts raw image bytes (JPEG, PNG, WebP) alongside a text prompt.
    Uses llama3.2-vision, LLaVA, or moondream depending on what is pulled.
    """

    name = "ollama_vision"
    description = (
        "Open-source vision-language model running locally via Ollama. "
        "Accepts image bytes + text prompt; returns text description."
    )
    modalities_in  = ["text", "image", "video", "bytes"]
    modalities_out = ["text"]

    def __init__(self, model_name: str | None = None):
        super().__init__(model_name)

    @property
    def active_model(self) -> str:
        if self._model_name:
            return self._model_name
        m = resolve_model(DEFAULT_VISION_PREFERENCE)
        if m is None:
            raise RuntimeError(
                "No vision-capable Ollama model available. "
                "Run: ollama pull llava  OR  ollama pull moondream"
            )
        return m


class OllamaEmbedTool(Tool):
    """
    Use Ollama's embedding endpoint to get high-dim embeddings from any
    open-source model, then project into the 8-dim Biblical space.

    Useful when the modality can be described in text but Word2Vec has
    poor coverage (e.g. technical jargon, foreign languages, code).
    """

    name = "ollama_embed"
    description = (
        "Embed text via a local Ollama model, then project into Biblical "
        "coordinate space.  Richer coverage than Word2Vec for out-of-vocab content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text":       {"type": "string"},
            "model_name": {"type": "string", "default": "nomic-embed-text"},
        },
        "required": ["text"],
    }
    modalities_in  = ["text"]
    modalities_out = ["bytes"]   # returns raw float32 embedding bytes

    def invoke(self, text: str, *, model_name: str = "nomic-embed-text") -> ToolResult:
        import ollama, struct
        try:
            resp = ollama.embeddings(model=model_name, prompt=text)
            vec = resp["embedding"]
            raw = struct.pack(f"{len(vec)}f", *vec)
            return ToolResult(
                content=vec,
                modality="bytes",
                raw_bytes=raw,
                source=f"ollama/{model_name}",
                metadata={"model": model_name, "dim": len(vec)},
            )
        except Exception as e:
            return ToolResult(
                content=[], modality="bytes", raw_bytes=b"",
                source=f"ollama/{model_name}",
                metadata={"error": str(e)},
            )
