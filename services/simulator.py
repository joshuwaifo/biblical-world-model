"""
Simulator: world-state at a canonical period.

Answers: "What does the graph look like when sliced to these books?"

Input:  a list of book abbreviations (defining the period), optional focus entity
Output: verse count, embedding centroid, top verses by in-period degree,
        high-betweenness verses, edge density, optional entity stats.
"""

from typing import Optional
import numpy as np
import networkx as nx
from services import store


def simulate(
    period_books: list[str],
    focus_entity: Optional[str] = None,
    top_k: int = 10,
) -> dict:
    # Resolve book abbreviations
    abbrs = [store.resolve_book_abbr(b) for b in period_books]
    verse_ids = store.verses_in_books(abbrs)
    if not verse_ids:
        raise ValueError(f"No verses found for books: {period_books}")

    verse_set = set(verse_ids)

    # Embedding centroid for the period
    vecs = np.stack([store.verse_embedding(vid) for vid in verse_ids])
    centroid = vecs.mean(axis=0).tolist()
    spread   = float(np.linalg.norm(vecs - vecs.mean(axis=0), axis=1).mean())

    # Build the in-period subgraph
    G = store.get_graph()
    sub_nodes = [("verse", vid) for vid in verse_ids]
    sub = G.subgraph(sub_nodes)

    density     = nx.density(sub)
    edge_count  = sub.number_of_edges()
    edge_types  = {}
    for _, _, d in sub.edges(data=True):
        et = d.get("edge_type", "?")
        edge_types[et] = edge_types.get(et, 0) + 1

    # Top verses by degree within the period subgraph
    degree_map = dict(sub.degree())
    top_by_degree = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_degree_out = []
    for node, deg in top_by_degree:
        if isinstance(node, tuple) and node[0] == "verse":
            m = store.get_meta(node[1])
            top_degree_out.append({"ref": m["ref"], "text": m["text"][:100], "degree": deg})

    # Betweenness on small subgraphs only (avoids O(VE) on large periods)
    betweenness_out = []
    if len(verse_ids) <= 5000:
        bc = nx.betweenness_centrality(sub, normalized=True)
        top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for node, score in top_bc:
            if isinstance(node, tuple) and node[0] == "verse":
                m = store.get_meta(node[1])
                betweenness_out.append({
                    "ref": m["ref"],
                    "text": m["text"][:100],
                    "betweenness": round(score, 6),
                })

    result: dict = {
        "books":            abbrs,
        "verse_count":      len(verse_ids),
        "edge_count":       edge_count,
        "edge_types":       edge_types,
        "subgraph_density": round(density, 6),
        "embedding_centroid": [round(x, 5) for x in centroid],
        "embedding_spread":   round(spread, 5),
        "top_verses_by_degree":    top_degree_out,
        "high_betweenness_verses": betweenness_out,
    }

    # Optional: verses most similar to focus_entity string in embedding space
    if focus_entity:
        try:
            focus_vec = store.embed_text(focus_entity)
            sims = vecs @ focus_vec        # cosine (vecs are already normalised)
            top_idx = np.argsort(sims)[::-1][:top_k]
            focus_hits = []
            for i in top_idx:
                vid = verse_ids[i]
                m = store.get_meta(vid)
                focus_hits.append({
                    "ref": m["ref"], "text": m["text"][:120],
                    "sim_to_focus": round(float(sims[i]), 4),
                })
            result["focus_entity"] = focus_entity
            result["focus_verses"] = focus_hits
        except ValueError:
            result["focus_entity"] = focus_entity
            result["focus_verses"] = []

    return result
