"""
Build the Bible graph from the DB and persist edges.

Expansion order (edit ACTIVE_EDGE_TYPES in config.py):
  Step 0  ["PRECEDES"]        — sequential order; irreducible baseline
  Step 1  + "IN_CHAPTER"     — same-chapter grouping
  Step 2  + "MENTIONS"       — verse↔entity links (requires entity extraction)
  Step 3  + "QUOTES"         — n-gram overlap between verse pairs across books
  Step 4  + "ECHOES"         — TF-IDF cosine similarity across books

Each edge function returns [(src_type, src_id, dst_type, dst_id, edge_type, weight)].

Run:  python graph/builder.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sqlalchemy.orm import Session

from config import (
    ACTIVE_EDGE_TYPES, MODELS_DIR,
    QUOTES_NGRAM_THRESHOLD, ECHOES_COSINE_THRESHOLD, ECHOES_MIN_BOOK_DISTANCE,
)
from db.session import SessionLocal
from db.models import Book, Chapter, Verse, Entity, EntityMention, GraphEdge

GRAPH_PATH    = MODELS_DIR / "bible_graph.gpickle"
NODE_MAP_PATH = MODELS_DIR / "node_map.pkl"   # {("verse"|"entity", id): node_idx}

Edge = tuple[str, int, str, int, str, float]  # src_type, src_id, dst_type, dst_id, edge_type, weight


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_verses(db: Session) -> list[dict]:
    rows = (
        db.query(Verse.id, Verse.text, Verse.number,
                 Chapter.number.label("chapter"),
                 Book.abbreviation.label("book"),
                 Book.canonical_order.label("book_order"))
        .join(Chapter, Verse.chapter_id == Chapter.id)
        .join(Book, Chapter.book_id == Book.id)
        .order_by(Book.canonical_order, Chapter.number, Verse.number)
        .all()
    )
    return [r._asdict() for r in rows]


def _load_entities(db: Session) -> list[dict]:
    return [
        {"id": r.id, "name": r.canonical_form}
        for r in db.query(Entity.id, Entity.canonical_form).all()
    ]


def _load_mentions(db: Session) -> list[tuple[int, int]]:
    return [
        (r.entity_id, r.verse_id)
        for r in db.query(EntityMention.entity_id, EntityMention.verse_id).all()
    ]


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------

def _precedes_edges(verses: list[dict]) -> list[Edge]:
    return [
        ("verse", verses[i]["id"], "verse", verses[i+1]["id"], "PRECEDES", 1.0)
        for i in range(len(verses) - 1)
    ]


def _in_chapter_edges(verses: list[dict]) -> list[Edge]:
    from itertools import combinations
    by_chapter: dict[tuple, list[int]] = {}
    for v in verses:
        by_chapter.setdefault((v["book"], v["chapter"]), []).append(v["id"])
    edges: list[Edge] = []
    for ids in by_chapter.values():
        for a, b in combinations(ids, 2):
            edges.append(("verse", a, "verse", b, "IN_CHAPTER", 1.0))
    return edges


def _mentions_edges(mentions: list[tuple[int, int]]) -> list[Edge]:
    return [
        ("verse", verse_id, "entity", entity_id, "MENTIONS", 1.0)
        for entity_id, verse_id in mentions
    ]


def _quotes_edges(verses: list[dict]) -> list[Edge]:
    """Character 4-gram overlap — detects direct textual quotes across books.

    Block-processes rows to avoid materialising the full N×N similarity matrix.
    Each block is (BLOCK × N) float32 ≈ 150 MB at BLOCK=1000, N=39k.
    """
    texts = [v["text"] for v in verses]
    ids   = [v["id"]   for v in verses]
    books = [v["book"] for v in verses]
    n     = len(verses)

    vec = CountVectorizer(analyzer="char", ngram_range=(4, 4), binary=True)
    mat = vec.fit_transform(texts).astype(np.float32).tocsr()
    # L2-normalise rows so dot product = cosine similarity
    from sklearn.preprocessing import normalize
    mat_norm = normalize(mat, norm="l2")   # sparse (N, vocab)

    books_arr = np.array(books)
    ids_arr   = np.array(ids)

    BLOCK  = 500
    edges: list[Edge] = []
    for i_start in range(0, n, BLOCK):
        i_end  = min(i_start + BLOCK, n)
        block  = mat_norm[i_start:i_end]              # sparse (BLOCK, vocab)
        sims   = (block @ mat_norm.T).toarray()       # dense  (BLOCK, N)

        for local_i in range(i_end - i_start):
            i     = i_start + local_i
            row   = sims[local_i]
            # upper-triangle only, different books, above threshold
            candidates = np.where(row[i + 1:] >= QUOTES_NGRAM_THRESHOLD)[0]
            for rel_j in candidates:
                j = i + 1 + rel_j
                if books_arr[i] != books_arr[j]:
                    edges.append(("verse", int(ids_arr[i]),
                                  "verse", int(ids_arr[j]),
                                  "QUOTES", float(row[j])))
    return edges


def _echoes_edges(verses: list[dict]) -> list[Edge]:
    """TF-IDF cosine similarity — detects lexical echoes across books."""
    texts       = [v["text"]       for v in verses]
    ids         = [v["id"]         for v in verses]
    book_orders = [v["book_order"] for v in verses]

    # token_pattern=r"\S+" to match our whitespace-split tokenisation
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+")
    mat = vec.fit_transform(texts).astype(np.float32)

    # Block-process to avoid O(n²) dense matrix for large corpora
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sim = cos_sim(mat)

    edges: list[Edge] = []
    for i in range(len(verses)):
        for j in range(i + 1, len(verses)):
            if abs(book_orders[i] - book_orders[j]) < ECHOES_MIN_BOOK_DISTANCE:
                continue
            s = float(sim[i, j])
            if s >= ECHOES_COSINE_THRESHOLD:
                edges.append(("verse", ids[i], "verse", ids[j], "ECHOES", s))
    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build() -> nx.DiGraph:
    db: Session = SessionLocal()

    print("Loading data...")
    verses   = _load_verses(db)
    entities = _load_entities(db) if "MENTIONS" in ACTIVE_EDGE_TYPES else []
    mentions = _load_mentions(db) if "MENTIONS" in ACTIVE_EDGE_TYPES else []
    print(f"  {len(verses):,} verses  {len(entities):,} entities")

    all_edges: list[Edge] = []

    if "PRECEDES" in ACTIVE_EDGE_TYPES:
        e = _precedes_edges(verses)
        print(f"  PRECEDES:   {len(e):,} edges")
        all_edges.extend(e)

    if "IN_CHAPTER" in ACTIVE_EDGE_TYPES:
        e = _in_chapter_edges(verses)
        print(f"  IN_CHAPTER: {len(e):,} edges")
        all_edges.extend(e)

    if "MENTIONS" in ACTIVE_EDGE_TYPES:
        e = _mentions_edges(mentions)
        print(f"  MENTIONS:   {len(e):,} edges")
        all_edges.extend(e)

    if "QUOTES" in ACTIVE_EDGE_TYPES:
        print("  Computing QUOTES (4-gram overlap)...")
        e = _quotes_edges(verses)
        print(f"  QUOTES:     {len(e):,} edges")
        all_edges.extend(e)

    if "ECHOES" in ACTIVE_EDGE_TYPES:
        print("  Computing ECHOES (TF-IDF cosine)...")
        e = _echoes_edges(verses)
        print(f"  ECHOES:     {len(e):,} edges")
        all_edges.extend(e)

    # Persist to DB
    db.query(GraphEdge).delete()
    db.add_all([
        GraphEdge(src_type=st, src_id=si, dst_type=dt, dst_id=di,
                  edge_type=et, weight=w)
        for st, si, dt, di, et, w in all_edges
    ])
    db.commit()

    # Build NetworkX DiGraph
    G = nx.DiGraph()

    for v in verses:
        G.add_node(("verse", v["id"]),
                   book=v["book"], chapter=v["chapter"],
                   verse_num=v["number"], book_order=v["book_order"])
    for e in entities:
        G.add_node(("entity", e["id"]), name=e["name"])
    for st, si, dt, di, et, w in all_edges:
        G.add_edge((st, si), (dt, di), edge_type=et, weight=w)

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"\nGraph  → {GRAPH_PATH}")
    print(f"  nodes: {G.number_of_nodes():,}   edges: {G.number_of_edges():,}")

    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    with open(NODE_MAP_PATH, "wb") as f:
        pickle.dump(node_map, f)
    print(f"Node map → {NODE_MAP_PATH}")

    db.close()
    return G


if __name__ == "__main__":
    build()
