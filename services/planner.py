"""
Planner: geodesic path between two canonical points.

Answers: "What is the shortest thematic path from verse A to verse B?"

Path finding uses the undirected QUOTES+ECHOES graph (no PRECEDES — that
would just return the trivial sequential chain).  The returned path includes
the edge type traversed at each step, the cosine similarity between adjacent
verses in embedding space, and the number of testament transitions.
"""

import networkx as nx
import numpy as np
from services import store


def plan(source_ref: str, target_ref: str) -> dict:
    src_id = store.lookup_verse_id(source_ref)
    dst_id = store.lookup_verse_id(target_ref)

    if src_id == dst_id:
        m = store.get_meta(src_id)
        return {
            "source": _verse_stub(src_id),
            "target": _verse_stub(dst_id),
            "path_length": 0,
            "path": [_verse_stub(src_id)],
            "testament_transitions": 0,
            "embedding_cosine_src_dst": 1.0,
        }

    src_node = ("verse", src_id)
    dst_node = ("verse", dst_id)
    PG = store.get_plan_graph()

    if src_node not in PG or dst_node not in PG:
        missing = []
        if src_node not in PG: missing.append(source_ref)
        if dst_node not in PG: missing.append(target_ref)
        raise ValueError(
            f"No QUOTES/ECHOES edges for: {missing}. "
            "These verses are isolated in the thematic graph."
        )

    try:
        raw_path: list = nx.shortest_path(PG, src_node, dst_node)
    except nx.NetworkXNoPath:
        raise ValueError(
            f"No thematic path found between {source_ref!r} and {target_ref!r}. "
            "Try expanding the graph with more edge types."
        )

    # Build annotated path
    path_out = []
    testament_transitions = 0
    prev_testament = None

    for i, node in enumerate(raw_path):
        vid = node[1]
        m = store.get_meta(vid)
        entry: dict = {
            "ref":      m["ref"],
            "book":     m["book_name"],
            "testament": m["testament"],
            "text":     m["text"],
        }
        if i < len(raw_path) - 1:
            next_node = raw_path[i + 1]
            data = PG.get_edge_data(node, next_node) or {}
            entry["edge_to_next"] = data.get("edge_type", "?")
            entry["weight_to_next"] = round(float(data.get("weight", 1.0)), 4)

            # Cosine similarity between consecutive verse embeddings
            v1 = store.verse_embedding(vid)
            v2 = store.verse_embedding(next_node[1])
            entry["cosine_to_next"] = round(float(np.dot(v1, v2)), 4)

        if prev_testament and m["testament"] != prev_testament:
            testament_transitions += 1
        prev_testament = m["testament"]

        path_out.append(entry)

    src_vec = store.verse_embedding(src_id)
    dst_vec = store.verse_embedding(dst_id)
    cosine_src_dst = float(np.dot(src_vec, dst_vec))

    return {
        "source":       _verse_stub(src_id),
        "target":       _verse_stub(dst_id),
        "path_length":  len(raw_path) - 1,
        "path":         path_out,
        "testament_transitions": testament_transitions,
        "embedding_cosine_src_dst": round(cosine_src_dst, 4),
    }


def _verse_stub(verse_id: int) -> dict:
    m = store.get_meta(verse_id)
    return {"ref": m["ref"], "book": m["book_name"], "testament": m["testament"], "text": m["text"]}
