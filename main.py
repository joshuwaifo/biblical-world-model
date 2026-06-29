"""
Biblical World Model — Universal Information Space Navigator

Architecture
────────────
Training corpus  :  Ethiopian Orthodox Tewahedo canon (38,927 verses)
                    Self-supervised GNN (BibleGAT, masked reconstruction)
                    → 8-dim Biblical coordinate space (the learned manifold)

The Biblical manifold is the universal coordinate system.
Any stream of bits/bytes — text, image, audio, video, binary data,
foundation model outputs — is projected into this space and navigated
using the geometry learned from the canon.

Three uses of the same trained GNN weights (arxiv 2506.22084: transformers = GNNs)
────────────────────────────────────────────────────────────────────────────────────
  next_action     GNN attention weights as action probabilities
                  → selects which tool to invoke next

  alignment       Reconstruction likelihood (same objective as training)
                  → scores retrieved content for truth / consistency

  novelty         Flipped reconstruction head
                  → detects genuinely new information outside the manifold

Tool layer (open-source, local, no API keys required)
──────────────────────────────────────────────────────
  web_search      DuckDuckGo (no key)
  fetch           URL fetch + modality detection
  ollama          Any locally-pulled open-source model
                  (Gemma, Llama, EXAONE, Mistral, Phi, Qwen, ...)
  ollama_vision   Vision-capable models (LLaVA, moondream, llama3.2-vision)

Endpoints
─────────
  POST /render      Geometric neighbourhood of a verse / concept
  POST /simulate    World-state at a canonical slice of the information space
  POST /plan        Shortest thematic path between two concepts
  POST /score       Oracle scoring: alignment + novelty for any content
  POST /navigate    Full guided search: query → tools → synthesis
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from services import store
from routes import render, simulate, plan, score, navigate as navigate_route


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all artefacts once: FAISS, graph, W2V, GNN weights, oracle, embedder
    store.init()
    yield


app = FastAPI(
    title="Biblical World Model",
    description=(
        "Self-supervised geometric world model trained on the Ethiopian Orthodox canon. "
        "Uses the learned Biblical geometry to navigate any information space. "
        "All models open-source, running locally via Ollama."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(render.router)
app.include_router(simulate.router)
app.include_router(plan.router)
app.include_router(score.router)
app.include_router(navigate_route.router)


@app.get("/")
def root():
    from tools.model import _available_models
    from tools import registry
    return {
        "model":        "BibleGAT  layers=2  heads=1  dim=8",
        "corpus":       "Ethiopian Orthodox canon (WEB + Deuterocanon + 1 Enoch)",
        "verses":       38927,
        "graph_edges":  "PRECEDES + QUOTES + ECHOES",
        "oracle": {
            "next_action":       "GNN attention weights as action policy",
            "alignment":         "reconstruction likelihood (truth scoring)",
            "novelty":           "flipped reconstruction head (discovery)",
        },
        "local_models": _available_models(),
        "tools":        registry.available_names(),
        "endpoints": ["/render", "/simulate", "/plan", "/score", "/navigate"],
    }
