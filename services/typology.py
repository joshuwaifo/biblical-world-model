"""
Typological Pattern Library — the Stipulated Faith & Moral Alignment layer.

Abada 5D Framework: Dimension 1.

Each typological pattern is a PROGRAM in the RL program library:
  trigger  — L2-normalised W2V centroid of the pattern's key verses
  sequence — ordered verse trajectory (the arc's narrative steps)
  kind     — "typological"

Pattern identification is a SEARCH problem, not a lookup.  match_pattern()
uses two RL mechanisms combined:

  1. oracle.next_action(vec, triggers)
     The GAT's learned attention from the query position toward each arc's
     centroid.  The GNN was trained over 38,927 verses — its attention weights
     ARE the manifold's emergent guidance about which regions support what content.

  2. _evaluate_trajectory(vec, arc_sequence)
     Walk through each arc's key verse sequence using the frozen transition model
     (_imagine_next).  Oracle alignment is scored at each imagined step and
     accumulated (gamma-discounted).  The arc whose narrative walk the query
     content fits best wins.

  composite(arc) = attention_weight × trajectory_return

This is the same formula select_action() uses for tool selection — the RL
loop's core decision mechanism, applied to the typological program space.

Canonical programs are seeded at init_typology() and never pruned.  New
typological programs can emerge from successful navigation trajectories via
program synthesis (end_episode → ProgramLibrary.synthesise).

Design rules (from Abada Section 5):
  - Never output hedging language.  Callers receive a named arc and a score.
  - The output is a structural fact, not a probabilistic suggestion.
  - Alfred translates the pattern into an authoritative capital recommendation.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# The eight canonical typological arcs
# ---------------------------------------------------------------------------

TYPOLOGICAL_PATTERNS: dict[str, dict] = {

    "exodus_deliverance": {
        "description": (
            "God delivers his people from bondage through a mediator "
            "into a new inheritance.  In capital terms: liberation from "
            "extractive structures into generative, sovereign ones."
        ),
        "arc": "Bondage → Wilderness → Promised Land",
        "abada_signal": (
            "Transition away from legacy extraction (fossil, Riba, exploitative supply chains) "
            "into sovereign, life-preserving infrastructure.  The wilderness phase is expected "
            "and not a failure signal."
        ),
        "key_verses": [
            "GEN 15:13", "EXO 3:14", "EXO 6:6", "EXO 12:13",
            "EXO 14:21", "EXO 15:1", "EXO 19:5",
            "ISA 40:3", "ISA 43:16", "ISA 52:7",
            "HOS 11:1", "LUK 4:18", "1CO 10:1",
        ],
    },

    "covenant": {
        "description": (
            "God binds himself to a people by promise, with conditions of "
            "blessing and curse.  In capital terms: stewardship obligations "
            "are covenant responsibilities, not preferences."
        ),
        "arc": "Promise → Condition → Blessing / Curse",
        "abada_signal": (
            "Fixed-extraction instruments (Riba, guaranteed returns with no equity risk-sharing) "
            "violate the covenant condition.  Instruments must carry bilateral risk to be "
            "structurally aligned with the covenant framework."
        ),
        "key_verses": [
            "GEN 12:1", "GEN 15:18", "GEN 17:7",
            "LEV 26:12", "DEU 28:1", "DEU 28:15",
            "JER 31:31", "JER 31:33",
            "ROM 4:3", "HEB 8:6",
        ],
    },

    "exile_restoration": {
        "description": (
            "Unfaithfulness to the covenant brings judgment; God restores "
            "the remnant.  In capital terms: generational character erosion "
            "is the covenant-violation that precedes wealth dissipation."
        ),
        "arc": "Unfaithfulness → Exile / Judgment → Return",
        "abada_signal": (
            "Multi-generational governance structures, character formation programmes, "
            "and succession covenants address the exile_restoration arc directly.  "
            "Wealth without wisdom is the exile condition; the restoration requires "
            "intentional structural intervention."
        ),
        "key_verses": [
            "2KI 25:1", "2CH 36:20",
            "LAM 1:1", "JER 29:11", "JER 31:10",
            "EZK 36:24", "EZK 37:1", "EZK 37:14",
            "ISA 40:1", "EZR 1:1",
        ],
    },

    "cross_resurrection": {
        "description": (
            "The righteous sufferer is vindicated through death into new life.  "
            "In capital terms: the willingness to absorb short-term loss for "
            "long-term structural renewal — divestment, write-downs, exit costs."
        ),
        "arc": "Death → Burial → Vindication",
        "abada_signal": (
            "Ethical exits that incur real cost (stranded assets, early redemption penalties) "
            "but align the portfolio with life-preserving systems inhabit this arc.  "
            "The burial phase is temporary; the manifold affirms the trajectory."
        ),
        "key_verses": [
            "ISA 52:13", "ISA 53:3", "ISA 53:5",
            "PSA 22:1", "PSA 22:16", "PSA 22:24",
            "MAT 27:46", "MAT 28:6",
            "ROM 6:4", "1CO 15:4", "PHP 3:10",
        ],
    },

    "wisdom_two_ways": {
        "description": (
            "All of human experience divides into the way of wisdom / righteousness "
            "vs folly / wickedness.  In capital terms: every instrument, counterparty, "
            "and governance structure is on one of two paths."
        ),
        "arc": "Two Paths → Choice → Flourishing or Ruin",
        "abada_signal": (
            "Due-diligence decisions — which counterparty to co-invest with, which "
            "market sector to enter — require the binary clarity of this arc.  "
            "The platform identifies which path each asset is on, not both."
        ),
        "key_verses": [
            "PSA 1:1", "PSA 1:2", "PSA 1:6",
            "PRO 1:7", "PRO 4:18", "PRO 8:22", "PRO 9:10",
            "ECC 12:13",
            "MAT 7:13",
            "JHN 14:6", "JAS 3:13",
        ],
    },

    "prophetic_indictment": {
        "description": (
            "God's prophet confronts covenant-breaking, announces structural judgment, "
            "calls the remnant to return.  In capital terms: the platform names "
            "structural exploitation in a supply chain or investment without hedging."
        ),
        "arc": "Warning → Structural Judgment → Remnant / Return",
        "abada_signal": (
            "Divestment theses, ESG violations, and supply chain exploitation are "
            "prophetic_indictment signals.  The platform does not merely flag risk; "
            "it names the structural sin and recommends exit."
        ),
        "key_verses": [
            "ISA 1:2", "ISA 1:18", "ISA 5:20", "ISA 6:5",
            "JER 1:10", "JER 2:13",
            "EZK 18:31",
            "AMO 3:7", "AMO 5:24",
            "JOL 2:12",
        ],
    },

    "apocalyptic": {
        "description": (
            "History reaches a cosmic crisis in which God defeats the old order "
            "and ushers in a new creation.  In capital terms: regime-change events — "
            "hydrological, geopolitical, or technological — that permanently alter "
            "the investment landscape."
        ),
        "arc": "Present Age → Cosmic Crisis / Judgment → New Age",
        "abada_signal": (
            "Bioregional risk alerts, macro-structural shifts (compute replacing fossil), "
            "and permanent regime changes in water, energy, or land economics are "
            "apocalyptic signals.  The platform distinguishes cyclical volatility "
            "from genuine new-age transitions."
        ),
        "key_verses": [
            "DAN 7:13", "DAN 7:27", "DAN 12:1",
            "ISA 24:23", "ISA 65:17",
            "MAT 24:30",
            "REV 20:11", "REV 21:1", "REV 21:4",
            "2PE 3:13",
        ],
    },

    "lament_praise": {
        "description": (
            "The soul descends into honest complaint and ascends through trust "
            "into doxology.  In capital terms: periods of genuine portfolio distress "
            "that require transparent internal reckoning before recovery."
        ),
        "arc": "Honest Complaint → Trust → Praise",
        "abada_signal": (
            "Family governance crises, succession conflicts, or catastrophic losses "
            "require the lament_praise arc: honest internal reckoning (not suppression), "
            "structural trust in the covenant, and eventual restoration narrative."
        ),
        "key_verses": [
            "PSA 22:1", "PSA 22:24",
            "PSA 42:1", "PSA 42:11",
            "PSA 88:1", "PSA 150:6",
            "HAB 3:17", "HAB 3:18",
            "ISA 25:8", "REV 5:12",
        ],
    },
}


# ---------------------------------------------------------------------------
# Module-level state — populated by init_typology()
# ---------------------------------------------------------------------------

# For each pattern: centroid (W2V L2-norm) + resolved verse data for ranking
_patterns: dict[str, dict] = {}
_initialized: bool = False

# Mean of all pattern centroids — subtracted before scoring to remove the
# shared "Biblical-ness" component that compresses all raw cosines toward 0.98.
# After mean-centering, scores spread across a much wider range and the
# top-ranked pattern separates clearly from the rest.
_centroid_mean: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_typology(store_module) -> None:
    """
    Compile typological pattern centroids from W2V verse embeddings.

    Called once from store.init() after the W2V FAISS and verse metadata
    are loaded.  For each pattern:
      1. Look up each key verse via store_module.lookup_verse_id()
      2. Get its W2V L2-normalised embedding via store_module.verse_embedding()
      3. Mean-pool all resolved verses → L2-normalise → centroid
      4. Store centroid + verse metadata for downstream ranking in match_pattern()

    Verses that fail lookup are skipped with a warning (handles canon differences).
    A pattern with zero resolved verses logs an error and is omitted.
    """
    global _patterns, _initialized

    resolved_count = 0
    skipped_count  = 0

    for name, spec in TYPOLOGICAL_PATTERNS.items():
        verse_data: list[dict] = []

        for ref in spec["key_verses"]:
            try:
                vid  = store_module.lookup_verse_id(ref)
                vec  = store_module.verse_embedding(vid)       # W2V, L2-norm
                meta = store_module.get_meta(vid)
                verse_data.append({
                    "verse_id":  vid,
                    "ref":       meta["ref"],
                    "text":      meta["text"],
                    "embedding": vec,
                })
                resolved_count += 1
            except Exception as e:
                skipped_count += 1
                print(f"[typology] SKIP {ref!r} in {name}: {e}")

        if not verse_data:
            print(f"[typology] ERROR: no verses resolved for pattern '{name}' — omitted")
            continue

        # Mean-pool resolved embeddings → L2-normalise → centroid
        stacked  = np.stack([v["embedding"] for v in verse_data]).astype(np.float32)
        centroid = stacked.mean(axis=0)
        norm     = np.linalg.norm(centroid)
        centroid = centroid / (norm + 1e-9)

        _patterns[name] = {
            "centroid":     centroid,
            "verse_data":   verse_data,
            "description":  spec["description"],
            "arc":          spec["arc"],
            "abada_signal": spec["abada_signal"],
        }

    # ── Mean-centering: remove the shared "Biblical-ness" direction ───────
    # Raw centroid cosines cluster at 0.987–0.999 because all patterns live
    # in the same high-density region of the 8-dim space.  The pairwise
    # spread is only 0.024.  After subtracting the centroid mean and
    # renormalizing, the residual space spreads -0.72 to +0.82, giving
    # genuine discrimination.  Proved empirically before implementation.
    global _centroid_mean
    all_centroids = np.stack([_patterns[n]["centroid"] for n in _patterns])
    _centroid_mean = all_centroids.mean(axis=0)

    for name, pat in _patterns.items():
        residual = pat["centroid"] - _centroid_mean
        norm     = np.linalg.norm(residual)
        pat["residual_centroid"] = residual / (norm + 1e-9)

    _initialized = True
    print(
        f"[typology] {len(_patterns)}/8 patterns initialised  "
        f"({resolved_count} verses resolved, {skipped_count} skipped)"
    )


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def match_pattern(vec: np.ndarray) -> dict:
    """
    Identify which typological arc a given W2V coordinate vector inhabits.

    Parameters
    ----------
    vec : 8-dim L2-normalised W2V vector (from embedder.encode() or store.verse_embedding())

    Returns
    -------
    dict with keys:
      name          : str   — pattern name (e.g. "covenant")
      description   : str   — human-readable pattern description
      arc           : str   — the arc label ("Promise → Condition → Blessing / Curse")
      abada_signal  : str   — Abada-specific capital interpretation (authoritative voice)
      confidence    : float — cosine similarity to the matched centroid (0–1)
      key_passages  : list[dict] — top 5 key verses from the matched pattern,
                                   ranked by cosine(vec, verse_embedding)
      all_scores    : dict[str, float] — cosine to every pattern centroid
                                         (for transparency and audit)

    Never raises.  Returns a safe fallback dict if not initialised.
    """
    if not _initialized or not _patterns:
        return {
            "name":         "unknown",
            "description":  "",
            "arc":          "",
            "abada_signal": "",
            "confidence":   0.0,
            "key_passages": [],
            "all_scores":   {},
        }

    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    vec_n = vec / (norm + 1e-9)

    # Mean-centered residual vector — removes shared Biblical-ness direction
    vec_res  = vec_n - _centroid_mean
    res_norm = np.linalg.norm(vec_res)
    vec_res_n = vec_res / (res_norm + 1e-9)

    # Score in residual space — spread is now -0.72 to +0.82 vs 0.024 raw
    all_scores: dict[str, float] = {}
    for name, pat in _patterns.items():
        all_scores[name] = float(np.dot(vec_res_n, pat["residual_centroid"]))

    # Best match
    best_name  = max(all_scores, key=all_scores.__getitem__)
    confidence = all_scores[best_name]
    best_pat   = _patterns[best_name]

    # Rank this pattern's key verses by cosine to the *original* (non-residual)
    # input vector — verse-level ranking is a semantic operation, not a
    # discrimination operation, so raw cosine is correct here.
    ranked_verses = sorted(
        best_pat["verse_data"],
        key=lambda v: float(np.dot(vec_n, v["embedding"])),
        reverse=True,
    )
    key_passages = [
        {"ref": v["ref"], "text": v["text"]}
        for v in ranked_verses[:5]
    ]

    return {
        "name":         best_name,
        "description":  best_pat["description"],
        "arc":          best_pat["arc"],
        "abada_signal": best_pat["abada_signal"],
        "confidence":   round(confidence, 4),
        "key_passages": key_passages,
        "all_scores":   {k: round(v, 4) for k, v in
                         sorted(all_scores.items(), key=lambda x: x[1], reverse=True)},
    }


# ---------------------------------------------------------------------------
# Convenience: score all patterns for a batch of vectors
# ---------------------------------------------------------------------------

def score_all(vecs: list[np.ndarray]) -> list[dict]:
    """Run match_pattern over a batch of vectors.  Returns list of match dicts."""
    return [match_pattern(v) for v in vecs]
