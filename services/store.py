"""
Shared in-process store: loads all artefacts once and exposes query helpers.

Call init() once from the FastAPI lifespan hook before serving requests.
"""

import pickle
import re
from pathlib import Path
from typing import Optional

import faiss
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from sqlalchemy.orm import Session

from config import (MODELS_DIR, FAISS_INDEX_PATH, VERSE_ID_MAP_PATH,
                    W2V_FAISS_INDEX_PATH, W2V_VERSE_ID_MAP_PATH)
from db.session import SessionLocal
from db.models import Book, Chapter, Verse, GraphEdge

# ---------------------------------------------------------------------------
# Module-level state — populated by init()
# ---------------------------------------------------------------------------
_faiss_index: Optional[faiss.Index] = None    # GNN-space (oracle internal use)
_verse_id_list: list[int] = []                # GNN FAISS row → verse_id
_w2v_faiss_index: Optional[faiss.Index] = None  # W2V-space (semantic search)
_w2v_verse_id_list: list[int] = []           # W2V FAISS row → verse_id
_verse_meta: dict[int, dict] = {}            # verse_id → metadata dict
_book_lookup: dict[str, str] = {}            # normalised name/abbrev → DB abbreviation
_graph: Optional[nx.DiGraph] = None          # full graph (all edge types)
_plan_graph: Optional[nx.Graph] = None       # undirected QUOTES+ECHOES only (for pathfinding)
_w2v: Optional[Word2Vec] = None
_w2v_vecs: dict[int, np.ndarray] = {}        # verse_id → raw W2V vector
_embeddings: dict[int, np.ndarray] = {}      # verse_id → raw GNN vector


def init() -> None:
    global _faiss_index, _verse_id_list, _verse_meta, _book_lookup
    global _graph, _plan_graph, _w2v, _embeddings
    global _w2v_faiss_index, _w2v_verse_id_list, _w2v_vecs

    # ── GNN FAISS (oracle internal: alignment, novelty, next_action) ──
    _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(VERSE_ID_MAP_PATH, "rb") as f:
        _verse_id_list = pickle.load(f)

    # ── W2V FAISS (semantic search: user queries, navigator position) ──
    _w2v_faiss_index = faiss.read_index(str(W2V_FAISS_INDEX_PATH))
    with open(W2V_VERSE_ID_MAP_PATH, "rb") as f:
        _w2v_verse_id_list = pickle.load(f)

    # ── GNN embeddings ──
    with open(MODELS_DIR / "gnn_embeddings.pkl", "rb") as f:
        _embeddings = pickle.load(f)

    # ── W2V per-verse vectors ──
    with open(MODELS_DIR / "verse_vectors.pkl", "rb") as f:
        _w2v_vecs = pickle.load(f)

    # ── Verse metadata from DB ──
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(
                Verse.id, Verse.text, Verse.number.label("verse_num"),
                Chapter.number.label("chap_num"),
                Book.name.label("book_name"),
                Book.abbreviation.label("book_abbr"),
                Book.testament,
                Book.canonical_order,
            )
            .join(Chapter, Verse.chapter_id == Chapter.id)
            .join(Book, Chapter.book_id == Book.id)
            .all()
        )
        for r in rows:
            _verse_meta[r.id] = {
                "id":        r.id,
                "ref":       f"{r.book_abbr} {r.chap_num}:{r.verse_num}",
                "book_name": r.book_name,
                "book_abbr": r.book_abbr,
                "testament": r.testament,
                "chap":      r.chap_num,
                "verse_num": r.verse_num,
                "text":      r.text,
                "can_order": r.canonical_order,
            }

        # ── Book name lookup ──
        books = db.query(Book.name, Book.abbreviation).all()
        for name, abbr in books:
            _book_lookup[abbr.lower()] = abbr           # "gen" → "GEN"
            _book_lookup[name.lower()] = abbr           # "genesis" → "GEN"
            # no-space variant: "1 samuel" → "1samuel" → "1SA"
            _book_lookup[name.lower().replace(" ", "")] = abbr
    finally:
        db.close()

    # ── NetworkX graph ──
    with open(MODELS_DIR / "bible_graph.gpickle", "rb") as f:
        _graph = pickle.load(f)

    # Undirected QUOTES+ECHOES subgraph for path finding
    _plan_graph = nx.Graph()
    PLAN_TYPES = {"QUOTES", "ECHOES"}
    for src, dst, data in _graph.edges(data=True):
        if data.get("edge_type") in PLAN_TYPES:
            _plan_graph.add_edge(src, dst, **data)

    # ── Word2Vec (for text-query embedding) ──
    _w2v = Word2Vec.load(str(MODELS_DIR / "word2vec.model"))

    print(f"[store] loaded {len(_verse_meta):,} verses  "
          f"GNN_FAISS={_faiss_index.ntotal:,}  W2V_FAISS={_w2v_faiss_index.ntotal:,}  "
          f"graph={_graph.number_of_nodes():,}N/{_graph.number_of_edges():,}E  "
          f"plan_graph={_plan_graph.number_of_edges():,}E")

    # ── Oracle (same GNN weights — three uses; uses GNN FAISS internally) ──
    from services import oracle as _oracle
    _oracle.init(_embeddings, _verse_id_list, _faiss_index)

    # ── Modal embedder (uses W2V FAISS for semantic encoding) ──
    from services import embedder as _embedder
    _embedder._init(_w2v, _w2v_faiss_index, _w2v_verse_id_list, _embeddings)

    # ── Typological pattern library (Abada 5D Dimension 1: Faith & Moral Alignment) ──
    import sys as _sys
    from services import typology as _typology
    _typology.init_typology(_sys.modules[__name__])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ref(ref: str) -> tuple[str, int, int]:
    """Parse 'Genesis 1:1' or 'GEN 1:1' → (abbr, chapter, verse)."""
    ref = ref.strip()
    m = re.match(r'^(.+?)\s+(\d+):(\d+)$', ref)
    if not m:
        raise ValueError(f"Cannot parse reference: {ref!r}  "
                         "Expected format: 'Book Chapter:Verse'")
    raw_book, chap, vnum = m.group(1), int(m.group(2)), int(m.group(3))
    key = raw_book.lower().replace(" ", "")
    abbr = _book_lookup.get(key) or _book_lookup.get(raw_book.lower())
    if abbr is None:
        raise ValueError(f"Unknown book: {raw_book!r}")
    return abbr, chap, vnum


def lookup_verse_id(ref: str) -> int:
    """Return verse_id for a canonical reference string."""
    abbr, chap, vnum = _parse_ref(ref)
    for vid, meta in _verse_meta.items():
        if meta["book_abbr"] == abbr and meta["chap"] == chap and meta["verse_num"] == vnum:
            return vid
    raise ValueError(f"Verse not found: {ref!r}")


def embed_text(text: str) -> np.ndarray:
    """Mean-pool Word2Vec tokens → L2-normalised 8-dim float32 (W2V space)."""
    tokens = text.split()
    vecs = [_w2v.wv[t] for t in tokens if t in _w2v.wv]
    if not vecs:
        raise ValueError(f"No known tokens in query: {text!r}")
    vec = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def knn(vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return [(verse_id, cosine_sim)] for the k nearest verses in W2V space.

    W2V space encodes semantic co-occurrence — use this for user-facing search,
    navigator position tracking, and Biblical context lookup.
    """
    q = normalize(vec.reshape(1, -1).astype(np.float32), norm="l2")
    scores, indices = _w2v_faiss_index.search(q, k)
    return [(_w2v_verse_id_list[int(i)], float(s))
            for i, s in zip(indices[0], scores[0]) if i >= 0]


def knn_gnn(vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return [(verse_id, cosine_sim)] for the k nearest verses in GNN space.

    GNN space encodes structural graph topology — used by the oracle internally
    for alignment scoring and novelty detection.
    """
    q = normalize(vec.reshape(1, -1).astype(np.float32), norm="l2")
    scores, indices = _faiss_index.search(q, k)
    return [(_verse_id_list[int(i)], float(s))
            for i, s in zip(indices[0], scores[0]) if i >= 0]


def verse_embedding(verse_id: int) -> np.ndarray:
    """Return the L2-normalised W2V embedding for a verse (semantic space)."""
    vec = _w2v_vecs[verse_id].astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def verse_gnn_embedding(verse_id: int) -> np.ndarray:
    """Return the L2-normalised GNN embedding for a verse (structural space)."""
    vec = _embeddings[verse_id].astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def get_meta(verse_id: int) -> dict:
    return _verse_meta[verse_id]


def get_graph() -> nx.DiGraph:
    return _graph


def get_plan_graph() -> nx.Graph:
    return _plan_graph


def resolve_book_abbr(raw: str) -> str:
    """Resolve a user-supplied book name/abbreviation to DB abbreviation."""
    key = raw.lower().replace(" ", "")
    abbr = _book_lookup.get(key) or _book_lookup.get(raw.lower())
    if abbr is None:
        raise ValueError(f"Unknown book: {raw!r}")
    return abbr


def verses_in_books(abbrs: list[str]) -> list[int]:
    """Return all verse_ids belonging to the given book abbreviations."""
    abbr_set = set(abbrs)
    return [vid for vid, m in _verse_meta.items() if m["book_abbr"] in abbr_set]
