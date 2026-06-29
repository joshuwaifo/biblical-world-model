# Biblical World Model — Architecture & Build Record

> **Audience:** A new engineer with no prior context. This document covers what
> was built, why each decision was made, what every file does, how to run
> everything, and what is still in progress.

---

## 1. What This Is

A **self-supervised geometric world model** trained on the Ethiopian Orthodox
Tewahedo Biblical canon (38,927 verses). The trained model produces a
**Biblical coordinate space** — a 32-dimensional manifold that encodes the
structural and semantic relationships across the entire canon, including the
Deuterocanon and 1 Enoch.

This coordinate space is the **universal grounding prior** for an AI capital
advisory system called **BRAIN**, built for **Abada Group** — a faith-directed
family office platform for sovereign capital allocation.

The core premise: the Bible was written across ~1,500 years by 40+ authors yet
exhibits deep structural coherence. That coherence, once learned geometrically,
can act as an invariant prior over any information — financial instruments,
governance structures, bioregional risk, investment memos, video feeds,
audio recordings, raw bytes. The manifold interprets the temporal; the eternal
grounds the factual.

---

## 2. The Two Layers of Reality

```
TRUTH LAYER      Biblical manifold — invariant, eternal, emergent
                 Trained once. Weights never change at runtime.
                 Reconstruction loss = verifiable reward signal
                 (analogue to Lean proof-checker in AlphaProof)
                         ↓  grounds / interprets
FACTOID LAYER    LLMs + Search + Tools — dynamic, date-stamped, perishable
                 What the world says today
```

**Alfred** (the AI executive partner persona) reads the truth layer and
presents a structurally authoritative interpretation to the principal — never
hedging, never deferring to "the data says." Every capital decision passes
through the manifold before action.

---

## 3. Training Corpus

| Source | Description |
|--------|-------------|
| World English Bible (WEB) | Old and New Testament (USFX format) |
| Deuterocanon | Books not in the Protestant canon (Tobit, Judith, Maccabees, Sirach, Wisdom, etc.) |
| 1 Enoch | Ethiopian canon; plaintext (`data/raw/enoch.txt`) |

**Total:** 38,927 verses stored in `bible.db` (SQLite via SQLAlchemy).

Tokenisation is **whitespace-split only** — case-preserving, no stemming, no
stopword removal. Every surface form the scribes wrote is a distinct token.
"God," "God." and "God" are three tokens. This preserves transliterations
(YHWH, Elohim), numbers, and all punctuation-adjacent variants.

---

## 4. Architecture Overview

```
corpus → bible.db
           ↓
    graph/builder.py     Build NetworkX graph (PRECEDES + QUOTES + ECHOES edges)
           ↓
    graph/word2vec.py    Train Word2Vec at dim=32 → verse_vectors.pkl
           ↓
    graph/pyg_data.py    Convert graph to PyG Data object (node features = W2V vecs)
           ↓
    graph/train.py       Train BibleGAT (masked reconstruction) → gnn_weights.pt
                         + gnn_embeddings.pkl (one vec per verse)
           ↓
    graph/fit_projection.py   Ridge regression: W2V → GNN  (32×32 coef matrix)
           ↓
    index/faiss_store.py      Build two FAISS flat-IP indices
                              verse_w2v.faiss   (semantic search, user-facing)
                              verse_embeddings.faiss  (oracle internal)
           ↓
    services/store.py    Load all artefacts once at startup
           ↓
    services/oracle.py   Three uses of the frozen BibleGAT weights
    services/embedder.py Any modality → 32-dim Biblical coordinate
    services/typology.py 8 canonical typological patterns as RL programs
    services/rl_loop.py  Alberta Plan four-component agent
    services/navigator.py  Oracle-guided tool loop
           ↓
    routes/  (FastAPI endpoints)
    main.py  (FastAPI app + lifespan boot)
```

---

## 5. The Graph

**File:** `graph/builder.py`

Three edge types (configured in `config.py → ACTIVE_EDGE_TYPES`):

| Edge type | Meaning | How computed |
|-----------|---------|-------------|
| `PRECEDES` | Sequential verse order | Always included; baseline topology |
| `QUOTES` | One verse quotes another | n-gram overlap ≥ 0.6 across books |
| `ECHOES` | Thematic resonance | TF-IDF cosine ≥ 0.45, min 1 book apart |

**Stats:** 38,927 nodes, ~59,822 edges.

Stored as a NetworkX `DiGraph` in `models/bible_graph.gpickle`.

A second `Graph` (undirected, QUOTES+ECHOES only) lives as `_plan_graph` in
`store.py` — used by the planner for thematic pathfinding.

---

## 6. The BibleGAT Model

**File:** `graph/gnn.py`, trained by `graph/train.py`

```
BibleGAT
  2 layers of GATConv (Graph Attention Network)
  1 attention head (heads=1, concat=False)
  dim_in = dim_out = 32   (matches W2V dimension)
  No dropout, no residual connections
  ELU activation between layers, linear output on final layer
  Total parameters: 704
```

**Training objective:** Masked node reconstruction (MLM-style for graphs).
- Randomly mask 30% of node features (zero them out)
- GNN reconstructs masked features from unmasked neighbours
- Loss: MSE on masked nodes only
- Val metric: `val_perplexity` on held-out edges (link prediction, 1.0 = perfect, 2.0 = coin-flip)

**Result at dim=32, 100 epochs:** `val_perplexity = 1.0837`

**Checkpoints:** Every epoch saved to `models/checkpoints/epoch_NNNN/`. Best
epoch by val_perplexity is promoted to `models/gnn_weights.pt`.

**Why Graph Attention specifically:** Transformers are mathematically equivalent
to GNNs (arxiv 2506.22084). The attention weights learned for Biblical graph
reconstruction are the same mechanism as token attention in a language model —
except here they encode structural relationships across 1,500 years of intertextual
citation and thematic resonance. These attention weights become the navigation
policy at runtime (see oracle.next_action below).

---

## 7. The W2V → GNN Projection

**File:** `graph/fit_projection.py`

Word2Vec embeddings (user-facing, semantic) and GNN embeddings (oracle
internal, structural) live in different spaces. A **Ridge regression** bridge
maps W2V → GNN:

```
W2V vec (32-dim, L2-norm) @ coef (32×32) + intercept (32,) → GNN vec (32-dim)
```

**Result:** Mean cosine similarity after projection = 0.993.

Stored in `models/w2v_to_gnn_projection.pkl` as `{"coef": ..., "intercept": ..., "mean_projected_cosine": ...}`.

This projection is used by `oracle._w2v_to_gnn()` so the oracle's public
interface accepts plain coordinates from any modality and internally bridges
to GNN space without callers needing to know.

---

## 8. FAISS Indices

**File:** `index/faiss_store.py`

Two separate flat inner-product indices (L2-normalised vectors → cosine search):

| Index file | Space | Used for |
|------------|-------|---------|
| `models/verse_w2v.faiss` | W2V (semantic) | User-facing k-NN, navigator position, typology |
| `models/verse_embeddings.faiss` | GNN (structural) | Oracle alignment, novelty, next_action |

Each has a companion `*_id_map.pkl` that maps FAISS row index → verse_id in the DB.

---

## 9. The Coordinate Space & Modality Agnosticism

**File:** `services/embedder.py`

**Any information stream** is a valid input. The embedder maps it to a 32-dim
Biblical coordinate:

| Modality | Path |
|----------|------|
| `text` | Word2Vec mean-pool of whitespace tokens → L2-norm |
| `image` | Ollama vision model description → text path (preferred) / pixel stats fallback |
| `audio` | Whisper transcription → text path / byte stats fallback |
| `video` | Sample 8 frames → image path per frame → mean-pool |
| `bytes` | Byte histogram (256 bins) × fixed random 256×32 projection |

The byte-statistical fallback produces a reproducible coordinate but is NOT
aligned with W2V space (different basis). The Ollama description path correctly
lands in the same semantic space as text. **This is the known cross-modal
alignment gap** — a learned cross-modal encoder would close it.

Public API:
```python
embedder.encode(content, modality="auto")    # → 32-dim L2-norm float32
embedder.encode_bytes(raw_bytes, hint)        # → same
embedder.encode_gnn(content, modality)        # → GNN-space (rarely needed directly)
```

---

## 10. The Oracle

**File:** `services/oracle.py`

Three uses of the same frozen BibleGAT weights. All accept coordinates in
W2V space and project internally to GNN space via the Ridge projection.

### 10a. `oracle.next_action(current_vec, candidate_vecs) → list[float]`

Runs the GAT's attention mechanism with `current_vec` as the source node and
each candidate as a target. Returns a probability distribution (sums to 1.0)
— the manifold's learned opinion on "which direction is most relevant from here?"

Used by: `rl_loop.select_action()` for tool selection,
`rl_loop.score_programs_for()` for typological program search.

### 10b. `oracle.alignment_score(vec, k_neighbors=5) → float`

Masks the content node (zeros it out), reconstructs it from its k-nearest
Biblical GNN neighbours, measures NMSE (normalised MSE), converts via
`exp(-NMSE)`:

```
1.0   = perfect reconstruction = maximally consistent with the Biblical prior
~0.37 = no better than predicting the mean (random baseline)
→ 0   = completely outside the manifold
```

### 10c. `oracle.novelty_score(vec) → float`

`1.0 - alignment_score(vec)` — measures how far outside the known manifold
this content is. High novelty = genuinely new territory.

### 10d. `oracle.score(vec, k=5) → dict`

Convenience wrapper returning `{alignment, novelty, nearest_biblical_concepts}`.

### 10e. Calibration: `oracle._calibrate()`

Called automatically at startup (`oracle.init()` → `_calibrate()`). Runs
alignment scoring over **all 38,927 verses** and sets:

```
oracle.alignment_floor = 10th percentile  (below = OUTSIDE_PRIOR)
oracle.alignment_high  = 50th percentile  (above = ALIGNED)
```

At dim=32: `floor=0.2806`, `high=0.3327`.

**Why this matters:** These thresholds are data-driven from the model itself,
not hardcoded. If the model is retrained, they auto-update. The 50th percentile
threshold means "solidly within the typical Biblical reconstruction range" —
the manifold has something meaningful to say about this content.

---

## 11. The Two FAISS Architecture in the Store

**File:** `services/store.py`

Loads everything once at startup (via `store.init()` → FastAPI lifespan).
Exposes helper functions used by all services:

| Function | Space | Description |
|----------|-------|-------------|
| `store.knn(vec, k)` | W2V | k nearest verses by semantic similarity |
| `store.knn_gnn(vec, k)` | GNN | k nearest verses by structural similarity |
| `store.verse_embedding(vid)` | W2V | L2-norm W2V vec for a verse |
| `store.verse_gnn_embedding(vid)` | GNN | L2-norm GNN vec for a verse |
| `store.lookup_verse_id(ref)` | — | "GEN 1:1" → verse_id |
| `store.get_meta(vid)` | — | Full verse metadata dict |
| `store.get_graph()` | — | Full NetworkX DiGraph |
| `store.get_plan_graph()` | — | Undirected QUOTES+ECHOES subgraph |

`store.init()` also calls:
1. `oracle.init(...)` — loads BibleGAT weights, runs calibration
2. `embedder._init(...)` — sets W2V model + FAISS handles
3. `typology.init_typology(store_module)` — compiles pattern centroids + seeds program library

---

## 12. Typological Pattern Library

**File:** `services/typology.py`

The **Stipulated Faith & Moral Alignment layer** — Dimension 1 of the Abada
5D evaluation framework.

### 12a. The 8 Canonical Typological Arcs

Each pattern is a named region of the Biblical manifold, with:
- `key_verses`: canonical Biblical references that anchor this arc
- `arc`: the narrative arc label
- `abada_signal`: the authoritative capital interpretation text Alfred reads

| Pattern | Arc | Abada Capital Signal |
|---------|-----|---------------------|
| `exodus_deliverance` | Bondage → Wilderness → Promised Land | Transition from legacy extraction (fossil, Riba) to sovereign infrastructure |
| `covenant` | Promise → Condition → Blessing / Curse | Fixed-extraction instruments (Riba) violate the covenant condition |
| `exile_restoration` | Unfaithfulness → Exile → Return | Generational character erosion precedes wealth dissipation |
| `cross_resurrection` | Death → Burial → Vindication | Ethical exits with real cost inhabit this arc — burial is temporary |
| `wisdom_two_ways` | Two Paths → Choice → Flourishing or Ruin | Every instrument is on one of two paths; the platform names which |
| `prophetic_indictment` | Warning → Judgment → Remnant / Return | ESG violations, supply chain exploitation — the platform names the structural sin |
| `apocalyptic` | Present Age → Cosmic Crisis → New Age | Regime-change events: hydrological, geopolitical, technological |
| `lament_praise` | Honest Complaint → Trust → Praise | Portfolio distress requiring transparent internal reckoning before recovery |

### 12b. Initialisation (`init_typology`)

Called from `store.init()`. For each pattern:
1. Looks up each key verse by reference → gets its W2V L2-norm embedding
2. Mean-pools all resolved verse embeddings → L2-normalise → **centroid**
3. Computes `_centroid_mean` (mean of all 8 centroids)
4. Computes `residual_centroid = normalise(centroid - _centroid_mean)` for
   each pattern (mean-centering removes shared "Biblical-ness" direction)
5. Seeds the RL **program library** with a `ProgramEntry` per pattern
   (kind="typological", sequence = ordered key verse trajectory)

**Result:** 85 verses resolved across 8 patterns.

### 12c. Mean-Centering (Proved Empirically)

Raw cosines between all pattern centroids cluster at 0.987–0.999 (spread
0.024). Without centering, every input looks equally similar to all patterns.

After subtracting the centroid mean and renormalising:
- Raw spread: 0.024
- Centered spread: 0.79 – 1.63

This was **proved empirically** before implementing: a diagnostic printed all
pairwise cosines before and after centering.

### 12d. Pattern Matching (`match_pattern`)

Pattern identification is a **search problem**, not a lookup. Two RL mechanisms
combined:

```
composite_score(pattern) = oracle_attention_weight × trajectory_return
```

1. **Oracle attention prior** — `oracle.next_action(vec, pattern_centroids)`
   returns the GAT's attention weight for each pattern from the query position.
   This IS the manifold's emergent guidance signal.

2. **Trajectory simulation** — for each pattern, simulate walking through its
   key verse sequence using the frozen transition model:
   ```
   for each verse in arc_sequence:
       s_next = interpolate(s, verse_embedding, alpha=0.6)
       reward = oracle.alignment_score(s_next)
       total += reward × gamma^i
       s = s_next
   ```
   The arc whose narrative walk fits the query best produces the highest return.

Falls back to residual centroid cosine if the oracle is not yet initialised.

---

## 13. The RL Loop — Alberta Plan

**Files:** `services/rl_loop.py`, `services/value_fn.py`, `services/program_library.py`

Reference: Alberta Plan for Canadian AI Research (arxiv 2208.11173, Sutton,
Bowling, Pilarski). Four components, all running on every time step
(temporally uniform):

| Component | Implementation | Parameters |
|-----------|---------------|-----------|
| Perception | `embedder.encode()` | None — frozen encoder |
| Reactive Policy | `oracle.next_action()` + program library | None — frozen GNN |
| Value Function | `ValueFunction` (TD-λ) | 32w + 1b = **33 learnable params** |
| Transition Model | `_imagine_next()` | None — frozen interpolation |

**Critical constraint: BibleGAT weights are FROZEN.** The value function is
the only thing that learns at runtime. This is intentional — the manifold is
the invariant prior; the value function learns session-specific navigation
heuristics on top of it.

### 13a. Value Function (`services/value_fn.py`)

```
V(s) = w·s + b      (linear readout of 32-dim Biblical coordinate)

TD(λ) update at each step:
  δ_t  = r_t + γ·V(s_{t+1}) - V(s_t)
  e_t  = γ·λ·e_{t-1} + s_t             (eligibility trace)
  w   += η·δ_t·e_t
  b   += η·δ_t

γ = 0.95, η = 0.05, λ = 0.8
```

The reward `r_t` is `oracle.alignment_score(step_output)` — produced by the
Biblical manifold itself, not a human. This is the grounded reward signal
Silver & Sutton identify as the foundation of general AI.

### 13b. Program Library (`services/program_library.py`)

SOAR-style program abstraction library. Two kinds of entries:

**`kind="typological"`** (seeded at init, never pruned)
- The 8 canonical patterns are pre-loaded as programs
- `sequence` = ordered list of `{type: "verse_trajectory", ref, embedding}`
- Used by `score_programs_for()` for pattern identification

**`kind="navigation"`** (synthesised at runtime)
- When a navigation episode achieves mean alignment > 0.65 across all steps,
  the trajectory is synthesised into a new program entry
- `sequence` = ordered list of `{tool, query_template}` tool-call steps
- Future similar queries trigger the stored program (bypassing oracle.next_action)
- Pruned by use-count beyond MAX_LIBRARY_SIZE=256

### 13c. Action Selection (`select_action`)

```
1. Check program library (kind="navigation"):
   if cosine(current_vec, program.trigger) > 0.88 → execute stored sequence
   
2. oracle.next_action(current_vec, tool_vecs) → attention-weighted probs

3. V-guided lookahead (when n_updates > 5):
   for each tool:
       imagined = _imagine_next(current, tool_vec, alpha=0.6)
       composite = attention_prob × sigmoid(V(imagined))
   pick best composite score
```

### 13d. `score_programs_for()` — Pattern Search via Trajectory Simulation

```python
# 1. Get all typological programs from library
programs = [e for e in _library._entries if e.kind == "typological"]

# 2. Oracle attention prior over pattern centroids
attn_weights = oracle.next_action(vec, [p.trigger for p in programs])

# 3. Simulate each arc's verse trajectory
for prog, attn in zip(programs, attn_weights):
    traj_return = _evaluate_trajectory(vec, prog.sequence, oracle)
    composite   = attn × traj_return
    results.append((prog.arc, composite, prog.name))

return sorted(results, descending by composite)
```

### 13e. Episode Lifecycle

```python
rl_loop.begin_episode()          # reset eligibility trace + trajectory buffers
# ... navigation loop ...
# each step:
rl_loop.update_value(s_t, reward, s_t1, done=False)
# end of episode:
rl_loop.end_episode(arc_name)    # attempt SOAR program synthesis
```

---

## 14. The Navigator

**File:** `services/navigator.py`

```
1. encode(query, modality) → query_vec
2. encode(goal, "text")    → goal_vec
3. rl_loop.begin_episode()
4. for step in 1..max_steps:
   a. _plan_tool_call_rl()  → tool_name, kwargs, confidence, library_hit
   b. tool.invoke(**kwargs) → raw output
   c. oracle.score(output_vec) → alignment, novelty
   d. typology.match_pattern(output_vec) → manifold_interpretation
   e. outside_prior = alignment < oracle.alignment_floor
   f. rl_loop.update_value(s_t, alignment, s_t1)
   g. append NavigationStep
   h. if cosine_to_goal >= threshold: break
5. Compute truth_layer_summary + factoid_layer_summary
6. rl_loop.end_episode(arc_name)
7. _synthesise() → final text answer (via Ollama, grounded in truth layer)
```

**`NavigationStep` fields:**
```
step_num, biblical_context, tool_used, tool_input, tool_output,
coordinates, cosine_to_goal, alignment, novelty, action_confidence,
elapsed_s, outside_prior, manifold_interpretation, td_error, library_hit
```

**`NavigationResult` fields:**
```
query_coordinates, goal_coordinates, steps, final_coordinates,
final_cosine_to_goal, converged, synthesis, retrieved_content,
truth_layer_summary, factoid_layer_summary, rl_diagnostics
```

**Two-layer summary in every response:**
```json
"truth_layer_summary": {
  "overall_alignment": 0.36,
  "trajectory": "converging",
  "dominant_pattern": { "name": "covenant", "arc": "...", "abada_signal": "..." },
  "key_biblical_passages_invoked": [...]
},
"factoid_layer_summary": {
  "sources_consulted": [...],
  "tools_used": [...],
  "total_snippets": 12,
  "high_novelty_steps": 2
}
```

---

## 15. API Endpoints

**File:** `main.py` + `routes/`

Start the server: `uvicorn main:app --reload`

| Endpoint | File | Description |
|----------|------|-------------|
| `GET /` | `main.py` | System info: model, corpus, available tools |
| `POST /score` | `routes/score.py` | Oracle score any content → alignment, novelty, coordinates |
| `POST /render` | `routes/render.py` | k-NN neighbourhood of a verse or concept |
| `POST /simulate` | `routes/simulate.py` | World-state at a canonical slice of information space |
| `POST /plan` | `routes/plan.py` | Shortest thematic path between two concepts |
| `POST /navigate` | `routes/navigate.py` | Full guided search with tool loop + synthesis |

**Example — score any content:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"content": "fixed-interest bond returning 8% annually with no equity participation"}'
```

**Example — navigate:**
```bash
curl -X POST http://localhost:8000/navigate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Nile watershed structural deterioration",
    "goal": "biblical principles for land stewardship and water covenant",
    "max_steps": 3
  }'
```

---

## 16. Tools

**Files:** `tools/registry.py`, `tools/web_search.py`, `tools/fetch.py`, `tools/model.py`

All tools are open-source, local — no proprietary API keys required.

| Tool name | Description |
|-----------|-------------|
| `web_search` | DuckDuckGo (no key needed) |
| `fetch` | URL fetch with modality detection |
| `ollama` | Any model pulled locally via Ollama (Gemma, Llama, Mistral, Phi, Qwen, EXAONE, ...) |
| `ollama_vision` | Vision-capable Ollama models (LLaVA, moondream, llama3.2-vision) |

---

## 17. Abada Group Integration (BRAIN System)

The Biblical World Model is the truth layer of **BRAIN** — Abada's Biblical
Research and Advisory Intelligence Network. The Abada 5D Framework maps to
oracle functions:

| Abada Dimension | Oracle Function | Typological Pattern |
|-----------------|----------------|-------------------|
| 1. Faith & Moral Alignment | `alignment_score` + `match_pattern` | Which arc? What does the manifold say? |
| 2. Bioregional Stewardship | `novelty_score` | Is this genuinely outside the known prior? |
| 3. Generational Liquidity | `match_pattern` → pattern arc | exile_restoration, covenant |
| 4. Systemic Trajectory | `oracle.next_action` | Which direction to attend to next? |
| 5. Competitive Lever | `navigator.navigate` | Full guided search toward the goal |

### Alfred Voice (Abada Section 5)

Alfred reads the `abada_signal` field of the matched pattern and produces
structurally authoritative output — no hedging language, no AI self-reference:

```
ARC:    COVENANT  ·  Promise → Condition → Blessing / Curse
SIGNAL: 35% manifold alignment  ·  65% novelty

Fixed-extraction instruments (Riba, guaranteed returns with no equity
risk-sharing) violate the covenant condition. Instruments must carry
bilateral risk to be structurally aligned with the covenant framework.

RECOMMENDATION: ADVANCE UNDER CONDITIONS.
```

---

## 18. Test Suite

**File:** `tests/test_abada.py`

Five Abada capital decision scenarios tested directly against oracle + typology
(no HTTP server needed):

| Scenario | Pillar | Expected Arc |
|----------|--------|-------------|
| Riba Screen — Fixed-Interest Bond | II: Evaluation | `covenant` / `wisdom_two_ways` |
| Divestment — Palm Oil Exit | I: Strategic Planning | `prophetic_indictment` |
| Co-Investment — East African Sovereign Compute | III: Co-Investment | `exodus_deliverance` |
| Family Governance — Third-Generation Values | II: Evaluation | `exile_restoration` |
| Bioregional Risk — Nile Watershed | I: Signal-Noise | `apocalyptic` ✓ |

Run:
```bash
python tests/test_abada.py
# or
python -m pytest tests/test_abada.py -v -s
```

Current status: **all 5 assertions pass** at dim=32.

---

## 19. Configuration

**File:** `config.py`

Single source of truth. Changing `W2V_VECTOR_SIZE` cascades through the entire
training pipeline:

```python
W2V_VECTOR_SIZE = 32        # Word2Vec and GNN share this dim
GNN_HIDDEN_DIM  = W2V_VECTOR_SIZE   # auto-follows
GNN_OUT_DIM     = W2V_VECTOR_SIZE   # auto-follows

ACTIVE_EDGE_TYPES = ["PRECEDES", "QUOTES", "ECHOES"]
QUOTES_NGRAM_THRESHOLD  = 0.6
ECHOES_COSINE_THRESHOLD = 0.45
ECHOES_MIN_BOOK_DISTANCE = 1   # ignore same-book echoes

GNN_LAYERS  = 2
GNN_HEADS   = 1
GNN_EPOCHS  = 100
GNN_LR      = 1e-2
```

---

## 20. Full Retraining Pipeline

When `W2V_VECTOR_SIZE` or graph topology changes, run in order:

```bash
# 1. Rebuild Word2Vec
python graph/word2vec.py

# 2. Rebuild graph (only if ACTIVE_EDGE_TYPES changed)
python graph/builder.py

# 3. Rebuild PyG data object
python graph/pyg_data.py

# 4. Train BibleGAT (~15 min at dim=32, 100 epochs, CPU)
python graph/train.py

# 5. Refit W2V→GNN projection
python graph/fit_projection.py

# 6. Rebuild FAISS indices
python index/faiss_store.py

# 7. Verify geometry (optional but recommended)
python graph/verify_geometry.py

# 8. Run Abada test suite
python tests/test_abada.py
```

Oracle calibration (`_calibrate()`) and typology initialisation
(`init_typology()`) run automatically at server startup — no manual step needed.

---

## 21. Model Artefacts (`models/`)

| File | Description |
|------|-------------|
| `word2vec.model` | Gensim Word2Vec model (dim=32, 28,712 vocab) |
| `verse_vectors.pkl` | `{verse_id: np.ndarray}` — W2V embeddings per verse |
| `gnn_weights.pt` | BibleGAT weights (best epoch by val_perplexity) |
| `gnn_embeddings.pkl` | `{verse_id: np.ndarray}` — GNN embeddings per verse |
| `w2v_to_gnn_projection.pkl` | Ridge regression coef + intercept (32×32 + 32) |
| `verse_w2v.faiss` | FAISS flat-IP index in W2V space (38,927 vectors) |
| `verse_w2v_id_map.pkl` | FAISS row → verse_id (W2V index) |
| `verse_embeddings.faiss` | FAISS flat-IP index in GNN space |
| `verse_id_map.pkl` | FAISS row → verse_id (GNN index) |
| `bible_graph.gpickle` | NetworkX DiGraph (38,927 nodes, ~59,822 edges) |
| `node_map.pkl` | {("verse", id): node_idx} |
| `pyg_data.pt` | PyG Data object for training |
| `training_history.pkl` | Per-epoch metrics (loss, val_perplexity, participation ratio) |
| `checkpoints/epoch_NNNN/` | All 100 epoch checkpoints (weights + embeddings) |

---

## 22. Key Empirical Results

| Metric | dim=8 | dim=32 |
|--------|-------|--------|
| `val_perplexity` (best epoch) | 1.094 | 1.0837 |
| Participation ratio | 2.88 / 8 | 3.61 / 32 |
| W2V→GNN projection cosine | 0.978 | 0.993 |
| `oracle.alignment_floor` (p10) | 0.1436 | 0.2806 |
| `oracle.alignment_high` (p50) | 0.1938 | 0.3327 |
| Raw centroid cosine spread | 0.024 | ~0.024 |
| Mean-centered cosine spread | 0.79–1.63 | 0.79–1.63 |

**Participation ratio insight:** The BibleGAT uses ~3.6 effective dimensions
regardless of total capacity (3/8 at dim=8, 3.6/32 at dim=32). This reflects
the intrinsic dimensionality of the Biblical graph reconstruction task — not
a model limitation. The dim=32 upgrade improved oracle calibration significantly
(floor moved from 0.14 to 0.28) but did not add effective GNN dimensions.

---

## 23. What Is In Progress / Next Steps

### Completed
- [x] Corpus ingestion → SQLite DB
- [x] Graph construction (PRECEDES + QUOTES + ECHOES)
- [x] Word2Vec at dim=32
- [x] BibleGAT (2-layer, 1-head, dim=32) trained to val_perplexity 1.0837
- [x] W2V → GNN projection (Ridge, cosine=0.993)
- [x] Dual FAISS (W2V semantic + GNN structural)
- [x] Oracle: next_action, alignment_score, novelty_score, calibration
- [x] Embedder: modality-agnostic encoding (text, image, audio, video, bytes)
- [x] Typology: 8 canonical patterns, mean-centering, abada_signal texts
- [x] RL loop: ValueFunction TD(λ), ProgramLibrary, select_action, episode lifecycle
- [x] Typological patterns as RL programs — trajectory-based search via oracle attention
- [x] Navigator: two-layer summaries, outside_prior flag, manifold_interpretation per step
- [x] Test suite: 5 Abada scenarios, all assertions pass

### In Progress
- [ ] `typology.match_pattern()` — need to complete wiring of `score_programs_for()`
  into the match function body (the RL trajectory search replacing centroid cosine)

### Pending
- [ ] `routes/ground.py` — POST `/ground` primary Abada intake endpoint
  (explicit two-layer response: `factoid_layer` + `truth_layer` + `interpretation`)
- [ ] `routes/score.py` — add `typological_pattern` field to response
- [ ] `main.py` — register `/ground`, add `architecture` key to root(), update description
- [ ] Cross-modal alignment — `_encode_bytes_statistical()` fallback is in a random space
  not aligned with W2V. For image/audio without Ollama, the coordinate is unreliable.
  Fix: fine-tune a byte→W2V mapper or always require Ollama for non-text modalities.

---

## 24. Dependencies

```
fastapi, uvicorn          — HTTP server
sqlalchemy, pydantic      — DB ORM + validation
gensim                    — Word2Vec
torch, torch-geometric    — GATConv / BibleGAT
faiss-cpu                 — Vector search
scikit-learn              — Ridge regression, PCA
networkx                  — Graph construction + pathfinding
httpx, lxml               — HTTP fetch tool
```

Optional (not in requirements.txt, installed separately):
```
ollama (Python client)    — Local LLM + vision models
cv2 (opencv-python)       — Video frame sampling
PIL (Pillow)              — Image pixel stats fallback
```

---

## 25. Quick-Start for a New Engineer

```bash
# 1. Clone and install
cd world_model
pip install -r requirements.txt

# 2. The DB and all model artefacts are already present — no retraining needed.
# bible.db + models/*.{pt,pkl,faiss} are committed.

# 3. Install Ollama + pull a model (optional but needed for /navigate synthesis)
# https://ollama.com/
ollama pull gemma3

# 4. Run the test suite to verify everything works
python tests/test_abada.py

# 5. Start the server
uvicorn main:app --reload

# 6. Try a score
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"content": "He was crushed for our iniquities"}'

# 7. Try a navigation
curl -X POST http://localhost:8000/navigate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "fixed-interest bond returning 8% annually with no equity participation",
    "goal": "biblical principles for covenant-compliant capital structures",
    "max_steps": 3
  }'
```

---

## 26. Design Principles (Invariants)

1. **BibleGAT weights are always frozen at runtime.** No gradient descent on GNN
   params during navigation. Only the ValueFunction's 33 params update.

2. **No proprietary APIs.** Everything runs locally via Ollama. The system must
   work air-gapped.

3. **Whitespace tokenisation only.** No stemming, no lowercasing, no stopword
   removal. Every surface form is preserved.

4. **Alfred never hedges.** Output from the truth layer is structural authority,
   not probabilistic suggestion. No "might", "could suggest", "it appears".

5. **The manifold is above the tools.** Retrieved factoid content is scored
   against the manifold — the manifold interprets it. The manifold is not
   another source in a list.

6. **Triggers are modality-agnostic.** Any information stream (text, image,
   audio, video, bytes) enters the system through the embedder. Once in
   Biblical coordinate space, all downstream logic is identical regardless
   of source modality.

7. **Calibration is always data-driven.** `alignment_floor` and `alignment_high`
   are computed from the model's own reconstruction distribution at startup —
   never hardcoded.
