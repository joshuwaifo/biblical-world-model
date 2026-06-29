from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "bible.db"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Word2Vec — minimum geometry is 2D (can be visualised directly).
# Expand geometrically when 2D stops separating meaningful clusters: 2→4→8→...
# Window=1: only immediate textual neighbours; the most local, least-assumption signal.
# ---------------------------------------------------------------------------
W2V_VECTOR_SIZE = 32
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_EPOCHS = 5

# ---------------------------------------------------------------------------
# Graph edges — start with the single most objective edge type (sequential
# order), add others only after inspecting the geometry at the previous step.
#
# Expansion order:
#   ["PRECEDES"]
#   + ["IN_CHAPTER"]
#   + ["MENTIONS"]
#   + ["QUOTES"]
#   + ["ECHOES"]
# ---------------------------------------------------------------------------
ACTIVE_EDGE_TYPES = ["PRECEDES", "QUOTES", "ECHOES"]

QUOTES_NGRAM_THRESHOLD = 0.6    # n-gram overlap fraction for a QUOTES edge
ECHOES_COSINE_THRESHOLD = 0.45  # TF-IDF cosine threshold for an ECHOES edge
ECHOES_MIN_BOOK_DISTANCE = 1    # ignore same-book echoes (too cheap)

# ---------------------------------------------------------------------------
# GNN — most irreducible: 1 layer, 1 attention head, same dim as input (no
# projection). No dropout until we have a reason to regularise.
# ---------------------------------------------------------------------------
GNN_LAYERS = 2
GNN_HEADS = 1
GNN_HIDDEN_DIM = W2V_VECTOR_SIZE   # no expansion; match input dim
GNN_OUT_DIM = W2V_VECTOR_SIZE
GNN_DROPOUT = 0.0
GNN_EPOCHS = 100
GNN_LR = 1e-2

# ---------------------------------------------------------------------------
# Poincaré ball — minimum hyperbolic geometry is 2D (same as W2V start).
# ---------------------------------------------------------------------------
POINCARE_DIM = 2
POINCARE_EPOCHS = 1
POINCARE_LR = 1e-2

# ---------------------------------------------------------------------------
# FAISS / retrieval
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH = MODELS_DIR / "verse_embeddings.faiss"   # GNN-space (oracle)
VERSE_ID_MAP_PATH = MODELS_DIR / "verse_id_map.pkl"

W2V_FAISS_INDEX_PATH = MODELS_DIR / "verse_w2v.faiss"       # W2V-space (semantic search)
W2V_VERSE_ID_MAP_PATH = MODELS_DIR / "verse_w2v_id_map.pkl"
