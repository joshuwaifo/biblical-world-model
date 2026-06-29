"""
Renderer: geometric neighbourhood of a verse.

Answers: "What does the canonical geometry say about this region?"

Input:  a verse reference ("Isaiah 53:5") OR a free-text query
Output: the query verse, its k-nearest geometric neighbours (ranked by cosine
        similarity), edge-type context from the graph, and the local subgraph.
"""

from typing import Optional
from services import store


def _edge_type_between(src_id: int, dst_id: int) -> Optional[str]:
    G = store.get_graph()
    src_node = ("verse", src_id)
    dst_node = ("verse", dst_id)
    data = G.get_edge_data(src_node, dst_node) or G.get_edge_data(dst_node, src_node)
    if data:
        return data.get("edge_type")
    return None


def _local_subgraph(verse_id: int, neighbor_ids: list[int]) -> dict:
    G = store.get_graph()
    node_set = {verse_id} | set(neighbor_ids)
    nodes = []
    for vid in node_set:
        m = store.get_meta(vid)
        nodes.append({"id": vid, "ref": m["ref"], "text": m["text"][:120]})
    edges = []
    seen = set()
    for src in node_set:
        for dst in node_set:
            if src == dst:
                continue
            key = (min(src, dst), max(src, dst))
            if key in seen:
                continue
            data = G.get_edge_data(("verse", src), ("verse", dst))
            if data:
                edges.append({
                    "src": src, "dst": dst,
                    "type": data.get("edge_type", "?"),
                    "weight": round(float(data.get("weight", 1.0)), 4),
                })
                seen.add(key)
    return {"nodes": nodes, "edges": edges}


def render_by_reference(ref: str, top_k: int = 10) -> dict:
    verse_id = store.lookup_verse_id(ref)
    vec = store.verse_embedding(verse_id)
    neighbors = store.knn(vec, top_k + 1)   # +1 because the verse itself is rank-0
    neighbors = [(vid, s) for vid, s in neighbors if vid != verse_id][:top_k]
    return _build_response(verse_id, neighbors)


def render_by_query(text: str, top_k: int = 10) -> dict:
    vec = store.embed_text(text)
    neighbors = store.knn(vec, top_k)
    # Top hit becomes the "anchor" verse; rest are neighbours
    if not neighbors:
        raise ValueError("No results for query")
    anchor_id = neighbors[0][0]
    rest = neighbors[1:]
    return _build_response(anchor_id, rest)


def _build_response(anchor_id: int, neighbors: list[tuple[int, float]]) -> dict:
    anchor_meta = store.get_meta(anchor_id)
    neighbor_ids = [vid for vid, _ in neighbors]

    geometric_neighbors = []
    for vid, sim in neighbors:
        m = store.get_meta(vid)
        etype = _edge_type_between(anchor_id, vid)
        geometric_neighbors.append({
            "ref":        m["ref"],
            "book":       m["book_name"],
            "testament":  m["testament"],
            "text":       m["text"],
            "cosine_sim": round(sim, 4),
            "edge_type":  etype,
        })

    return {
        "query_verse": {
            "id":       anchor_id,
            "ref":      anchor_meta["ref"],
            "book":     anchor_meta["book_name"],
            "testament":anchor_meta["testament"],
            "text":     anchor_meta["text"],
        },
        "geometric_neighbors": geometric_neighbors,
        "local_subgraph": _local_subgraph(anchor_id, neighbor_ids),
    }
