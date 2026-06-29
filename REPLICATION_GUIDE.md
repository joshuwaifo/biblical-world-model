# Biblical World Model — Replication Guide

**From blank canvas to a running system. No assumptions. No proprietary APIs.**

This guide reproduces every step taken to build the Biblical World Model: a
self-supervised Graph Attention Network trained on the Ethiopian Orthodox canon
that serves as a universal grounding prior for any information stream.

---

## 0. Why the Bible?

This is not a stylistic choice. It is an architectural one.

**The Bible is the densest cross-referenced text in human history.**
Written across 1,500 years by 40+ authors in three languages on three
continents, it converges on a single unified theological arc. That convergence
is not asserted — it is statistically observable. The QUOTES and ECHOES edges
in the graph (built in Step 5) are computed from raw character n-gram overlap
and TF-IDF cosine similarity, with no theological annotation. The intertextual
structure emerges from the text itself.

**This makes it self-supervised in a deep sense.** The corpus does not need
labels. The graph structure IS the signal. When the GNN learns to reconstruct
masked verse features from its neighbours, it is learning the geometry of that
1,500-year convergence — not what a human annotator decided it meant.

**Mathematical analogy (AlphaProof):**

```
AlphaProof        Biblical World Model
─────────────     ──────────────────────────────────────
Lean verifier  ←→ Biblical manifold (frozen GNN weights)
Proof closes   ←→ alignment_score(content) ≥ alignment_high
Proof fails    ←→ alignment_score(content) < alignment_floor
Novel theorem  ←→ novelty_score(content) → 1.0
```

The reconstruction loss is verifiable. When content scores high on
`alignment_score`, it means the GNN can reconstruct it from the Biblical
graph's neighbourhood structure. That is a mathematical statement, not a
theological one — though it happens to have deep theological interpretation.

**Eight canonical typological arcs are the natural coordinate axes.**
They are not invented or imposed. They emerge from which verse clusters
separate in the trained GNN space:

| Arc | Capital interpretation |
|-----|----------------------|
| Exodus / Deliverance | Liberation from extractive structures |
| Covenant | Stewardship obligations; bilateral risk required |
| Exile / Restoration | Generational character erosion as covenant failure |
| Cross / Resurrection | Absorb short-term loss for structural renewal |
| Wisdom: Two Ways | Every instrument is on one of two paths |
| Prophetic Indictment | Name structural sin; recommend exit |
| Apocalyptic | Permanent regime changes vs cyclical volatility |
| Lament / Praise | Honest reckoning before recovery |

**All source texts are public domain.** No licence fee. No API key.
No permission required to reproduce.

---

## 1. Prerequisites

### Hardware
- RAM: 8 GB minimum, 16 GB recommended
- CPU: Any modern multi-core (4+ cores). GPU optional — CPU training of
  BibleGAT over 38k nodes completes in ~30 minutes.
- Disk: 500 MB for trained artefacts; ~15 MB for raw corpus downloads

### Software
```bash
# OS: Ubuntu 22.04+ (macOS works; Windows untested)
python --version          # need 3.11+
git --version             # any recent version

# Install Ollama (local open-source LLM inference — no API key required)
curl -fsSL https://ollama.ai/install.sh | sh
ollama --version          # verify
```

### Ollama models (optional — system works without them; interpretation will be null)
```bash
ollama pull llama3.2            # Alfred interpretation text
ollama pull llava               # image → description → Biblical coordinates
# Any model works — list available: ollama list
```

---

## 2. Project Setup

### Directory layout
```
world_model/
├── config.py               # Single source of truth for all hyperparameters
├── main.py                 # FastAPI app — lifespan hook loads all artefacts
├── requirements.txt        # Pinned dependencies
├── bible.db                # SQLite — 38,927 verses (created in Step 3)
│
├── data/
│   ├── ingest.py           # Downloads + parses corpus into SQLite
│   └── raw/                # Cached downloads (web_usfx.zip, enoch.txt)
│
├── db/
│   ├── models.py           # SQLAlchemy ORM: Book, Chapter, Verse, GraphEdge
│   └── session.py          # Engine + SessionLocal factory
│
├── graph/
│   ├── word2vec.py         # Train Word2Vec on Biblical corpus
│   ├── builder.py          # Build Bible graph (PRECEDES, QUOTES, ECHOES)
│   ├── pyg_data.py         # NetworkX → PyTorch Geometric Data object
│   ├── gnn.py              # BibleGAT architecture definition
│   ├── train.py            # Self-supervised GNN training
│   ├── fit_projection.py   # W2V → GNN Ridge regression projection
│   └── verify_geometry.py  # W2V geometry validation
│
├── index/
│   └── faiss_store.py      # Build + persist both FAISS indices
│
├── services/
│   ├── store.py            # Shared in-process store — loads all artefacts
│   ├── oracle.py           # Three uses of BibleGAT (policy / alignment / novelty)
│   ├── embedder.py         # Modality-agnostic encoder (text/image/audio/video/bytes)
│   ├── typology.py         # Eight canonical typological arc programs
│   ├── rl_loop.py          # Alberta Plan RL loop (TD-λ, program library)
│   ├── value_fn.py         # TD(λ) value function (33 params, updates every step)
│   ├── program_library.py  # SOAR-style program synthesis + retrieval
│   ├── navigator.py        # Guided search agent (query → tools → synthesis)
│   ├── planner.py          # Geodesic path between two concepts
│   ├── renderer.py         # Geometric neighbourhood of a verse/concept
│   └── simulator.py        # World-state at a canonical period
│
├── routes/
│   ├── render.py           # POST /render
│   ├── simulate.py         # POST /simulate
│   ├── plan.py             # POST /plan
│   ├── score.py            # POST /score
│   ├── navigate.py         # POST /navigate
│   └── ground.py           # POST /ground  (two-layer truth/factoid response)
│
├── tools/
│   ├── base.py             # BaseTool interface
│   ├── web_search.py       # DuckDuckGo (no API key)
│   ├── fetch.py            # URL fetch + modality detection
│   ├── model.py            # Ollama local LLM/vision tool
│   └── registry.py         # Tool registry (lazy load)
│
├── tests/
│   └── test_abada.py       # 5 Abada capital decision integration tests
│
└── models/                 # All trained artefacts (created by pipeline below)
    ├── word2vec.model
    ├── verse_vectors.pkl
    ├── bible_graph.gpickle
    ├── node_map.pkl
    ├── pyg_data.pt
    ├── verse_id_to_idx.pkl
    ├── gnn_weights.pt
    ├── gnn_embeddings.pkl
    ├── training_history.pkl
    ├── verse_embeddings.faiss
    ├── verse_id_map.pkl
    ├── verse_w2v.faiss
    ├── verse_w2v_id_map.pkl
    ├── w2v_to_gnn_projection.pkl
    └── checkpoints/         # Per-epoch snapshots during GNN training
```

### Python environment
```bash
cd world_model
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### requirements.txt (exact pinned versions)
```
fastapi==0.115.6
uvicorn[standard]==0.32.1
sqlalchemy==2.0.36
pydantic==2.10.3
pydantic-settings==2.7.0

# NLP / text processing
gensim==4.3.3
spacy==3.8.3
scikit-learn==1.6.0
nltk==3.9.1

# Geometric Deep Learning
torch==2.5.1
torch-geometric==2.6.1
geoopt==0.5.0
networkx==3.4.2

# Vector search
faiss-cpu==1.9.0.post1

# HTTP / data fetching
httpx==0.28.1
lxml==5.3.0
```

### Verify install
```bash
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_geometric; print('pyg', torch_geometric.__version__)"
python -c "import faiss; print('faiss ntotal test:', faiss.IndexFlatIP(8).ntotal)"
python -c "import gensim; print('gensim', gensim.__version__)"
```

---

## 3. Corpus — Ingest the Ethiopian Orthodox Canon

### What gets ingested

| Source | Books | Verses | Testament | Licence |
|--------|-------|--------|-----------|---------|
| World English Bible British Edition | 66 | ~31,102 | OT + NT | Public domain (eBible.org) |
| WEB Deuterocanon | 15 | ~5,700 | Deuterocanon | Public domain (eBible.org) |
| 1 Enoch (R.H. Charles 1917) | 1 | ~1,061 | Ethiopian | Public domain (Gutenberg #77935) |
| **Total** | **82** | **~38,927** | — | — |

**Why 1 Enoch?** The Ethiopian Orthodox Tewahedo canon includes it as Scripture.
It is directly quoted in the Epistle of Jude and extensively echoed in Revelation.
Excluding it truncates the QUOTES/ECHOES edge graph and misses the apocalyptic
intertextual cluster that is critical for the `apocalyptic` typological arc.

### Run ingestion
```bash
python data/ingest.py
```

The script downloads automatically. Safe to re-run — cached in `data/raw/`.

Expected output:
```
WEB British Edition + Deuterocanon...
  GET     https://eBible.org/Scriptures/eng-webbe_usfx.zip
  36,802 verses parsed
1 Enoch (R.H. Charles 1917 ed.)...
  GET     https://www.gutenberg.org/cache/epub/77935/pg77935.txt
  1,061 verses parsed

Total verses in DB: 38,927 (or near — exact count varies by USFX revision)
```

### Verify
```bash
sqlite3 bible.db "SELECT COUNT(*) FROM verse;"
# → 38927

sqlite3 bible.db "
  SELECT b.testament, COUNT(v.id) AS n
  FROM verse v
  JOIN chapter c ON v.chapter_id = c.id
  JOIN book b ON c.book_id = b.id
  GROUP BY b.testament;
"
# → OT | ~23,145
#   NT | ~7,957
#   Deuterocanon | ~5,764
#   Ethiopian | ~1,061
```

### Database schema
```sql
Book    (id, name, abbreviation, testament, canonical_order)
Chapter (id, book_id, number)
Verse   (id, chapter_id, number, text)
GraphEdge (id, src_type, src_id, dst_type, dst_id, edge_type, weight)
```

---

## 4. Word2Vec — Biblical Vocabulary Geometry

### Why Word2Vec before the GNN

The GNN requires node feature vectors. Word2Vec provides them from raw
co-occurrence statistics in the Biblical text, with no labels or annotations.

The pipeline order is therefore: text → Word2Vec → GNN (not the reverse).

### Critical tokenisation rule

**Whitespace split only. No stemming. No lowercasing. No punctuation removal.**

This preserves:
- Case distinctions: `LORD` (YHWH) vs `lord` (human authority)
- Hebrew/Greek transliterations: `YHWH`, `Yeshua`, `Adonai`, `Elohim`
- Proper nouns as single tokens: `Israel`, `Jerusalem`, `Abaddon`
- Verse-boundary punctuation as separate tokens (contributes to positional encoding)

Any normalisation at this stage destroys information that the GNN will later
need to reconstruct.

### Configuration (`config.py`)
```python
W2V_VECTOR_SIZE = 32    # embedding dimension
W2V_WINDOW      = 5     # context window (immediate neighbours)
W2V_MIN_COUNT   = 1     # keep ALL tokens — every word matters
W2V_EPOCHS      = 5     # 5 passes through the full corpus
```

**On dimensionality:** The system started at 8-dim and was upgraded to 32-dim.
At 8-dim, only ~3 effective dimensions were active (participation ratio ≈ 3/8).
At 32-dim, ~12–16 effective dimensions allow the 8 typological patterns to
separate clearly. `W2V_VECTOR_SIZE` is the single config constant that
propagates to Word2Vec, BibleGAT, both FAISS indices, and the projection matrix
— change it once and re-run the pipeline.

### Run
```bash
python graph/word2vec.py
```

Expected output:
```
Training Word2Vec dim=32 window=5 min_count=1 epochs=5 ...
Saving models/word2vec.model
Computing verse vectors for 38927 verses...
Saving models/verse_vectors.pkl
[spot checks]
  "God"       → ['LORD', 'God', 'said', 'Israel', ...]
  "covenant"  → ['promise', 'oath', 'LORD', 'Israel', ...]
  "Jesus"     → ['Christ', 'Lord', 'disciples', 'said', ...]
```

Outputs:
- `models/word2vec.model` — Gensim Word2Vec model (~8 MB)
- `models/verse_vectors.pkl` — `{verse_id: np.ndarray(32,)}` for all verses

### Validate geometry (optional but recommended)
```bash
python graph/verify_geometry.py
```

Outputs:
- `models/geometry_report.txt` — testament separation, axis variance, NN coherence
- `models/geometry_check.png` — 2D PCA scatter coloured by testament

**What to look for:**
- OT and NT centroids should be separated (different vocabulary density)
- No axis should have near-zero variance (collapsed dimension = wasted capacity)
- Same-book rate in top-5 nearest neighbours should be > 60% (coherence check)

If geometry looks collapsed (all verses cluster at one point), reduce
`W2V_MIN_COUNT` or increase `W2V_EPOCHS`.

---

## 5. Bible Graph — The Intertextual Structure

### Why a graph, not a sequence

The Bible is not a linear text. It is a directed graph of textual relationships
built across 40 authors and 1,500 years. Reading it as a sequence destroys the
information that makes it unique: Paul quotes Isaiah; Revelation echoes Ezekiel;
the Gospel of John echoes Genesis 1.

The graph makes these relationships computable.

### Three edge types

**PRECEDES** (backbone):
One directed edge from each verse to the next in canonical order.
This is the only structural assumption — narrative sequence.
Every other edge type is computed from text statistics alone.

**QUOTES** (direct quotation):
4-character n-gram overlap ≥ 0.6 between verses in *different books*.
Threshold: `QUOTES_NGRAM_THRESHOLD = 0.6` in `config.py`.
Captures: NT quotations of OT, direct textual echoes.
Example: Matthew 27:46 quoting Psalm 22:1 ("My God, my God, why...").

**ECHOES** (lexical resonance):
TF-IDF cosine similarity ≥ 0.45 between verses in *different books*,
minimum 1 book apart. Threshold: `ECHOES_COSINE_THRESHOLD = 0.45`.
Captures: thematic resonance without direct quotation.
Example: Isaiah's Servant Songs echoing Psalm 22 conceptually.

Same-book edges are excluded for QUOTES and ECHOES to prevent trivial
intra-chapter clustering from dominating the cross-canonical signal.

### Run
```bash
python graph/builder.py
```

Expected output:
```
Loading verses from DB...
Building PRECEDES edges: 38,926
Building QUOTES edges (ngram threshold=0.6): ~14,000
Building ECHOES edges (cosine threshold=0.45, min_book_dist=1): ~6,000
Total edges: ~59,822
Saving models/bible_graph.gpickle
Saving models/node_map.pkl
Writing edges to GraphEdge table...
Done.
```

### Verify
```bash
python -c "
import pickle
G = pickle.load(open('models/bible_graph.gpickle', 'rb'))
print(f'Nodes: {G.number_of_nodes():,}')
print(f'Edges: {G.number_of_edges():,}')
by_type = {}
for _, _, d in G.edges(data=True):
    et = d.get('edge_type', 'unknown')
    by_type[et] = by_type.get(et, 0) + 1
for et, n in sorted(by_type.items()):
    print(f'  {et}: {n:,}')
"
```

---

## 6. PyTorch Geometric Format

Convert the NetworkX graph to a PyTorch Geometric `Data` object for efficient
GPU/CPU message passing.

```bash
python graph/pyg_data.py
```

Expected output:
```
Loaded 38927 verse vectors (32-dim)
Built edge_index with N edges
Saved models/pyg_data.pt
Saved models/verse_id_to_idx.pkl
Data summary: Data(x=[38927, 32], edge_index=[2, N])
```

Verify:
```bash
python -c "
import torch
d = torch.load('models/pyg_data.pt')
print('nodes:', d.x.shape)          # [38927, 32]
print('edges:', d.edge_index.shape)  # [2, ~119644]
print('feature range:', d.x.min().item(), d.x.max().item())
"
```

---

## 7. Train BibleGAT — The Frozen Prior

This is the core of the system. Everything else is downstream of this one model.

### Architecture: BibleGAT

```
Input: 38,927 verse nodes × 32-dim Word2Vec features
       + edge_index from PRECEDES + QUOTES + ECHOES

Layer 1: GATConv(in=32, out=32, heads=1, concat=False)
          ELU activation
Layer 2: GATConv(in=32, out=32, heads=1, concat=False)
          (no activation on final layer)

Output: 38,927 × 32-dim structural embeddings

Total parameters: 704
```

**Why so few parameters?** Occam's razor applied rigorously. The GNN's job is
to learn the manifold's geometry, not to memorise content. Overfitting to
surface features is exactly what must be avoided — the model must generalise
to novel content (new capital decisions, new images, new audio) that was not
in the training corpus.

**Why 1 attention head?** Start from the minimum. Multiple heads can be added
if the geometry shows insufficient separation. At 32-dim with 1 head, the
geometry already achieves val_perplexity ≈ 1.08.

### Training signal: Masked Feature Reconstruction

```
For each training batch:
  1. Randomly select 15% of verse nodes
  2. Zero their feature vectors (mask them)
  3. Pass through BibleGAT: reconstruct features from unmasked neighbours
  4. Loss: MSE between predicted and actual features, on masked nodes only
  5. Backprop, update weights
```

This is the graph analogue of BERT's masked language modelling. The GNN learns
to reconstruct masked nodes from their graph neighbourhood — which means it
must learn what "neighbourhood" means in the Biblical intertextual structure.

**The training signal is entirely self-supervised.** No human labels. No
annotation. The corpus IS the teacher: the graph topology built from character
overlap and TF-IDF similarity is what the model learns to be consistent with.

### Run
```bash
python graph/train.py
```

Expected training output (100 epochs):
```
Epoch   1/100  train_loss=0.4821  val_perplexity=1.4302  participation=8.23
Epoch  10/100  train_loss=0.2341  val_perplexity=1.2104  participation=11.47
Epoch  50/100  train_loss=0.1102  val_perplexity=1.1203  participation=13.82
Epoch 100/100  train_loss=0.0891  val_perplexity=1.0837  participation=15.21
Best checkpoint: epoch 100  val_perplexity=1.0837
Saved models/gnn_weights.pt
Saved models/gnn_embeddings.pkl  (38927 embeddings)
Saved models/training_history.pkl
```

**Target: val_perplexity ≈ 1.08** (1.0 = perfect; e≈2.72 = mean-prediction baseline)

**Participation ratio** measures how many PCA dimensions are actively used.
At 32-dim, target ≈ 12–16 effective dimensions. If participation stays below 6,
the model may need more epochs or a slightly higher learning rate.

Training time: ~30 min on CPU, ~5 min on GPU.

### CRITICAL: the weights freeze here

After `graph/train.py` completes, the weights in `models/gnn_weights.pt` are
**permanently frozen**. No gradient descent ever runs on them again at runtime.

The Biblical manifold is fixed. All learning at runtime happens in the
TD(λ) value function (33 parameters), which adapts to individual query
trajectories without touching the base model.

This is the same principle as AlphaProof's Lean verifier: the verifier does
not learn; it judges. The GNN judges alignment.

### Verify
```bash
python -c "
import pickle, numpy as np
emb = pickle.load(open('models/gnn_embeddings.pkl', 'rb'))
vecs = np.stack(list(emb.values()))
print(f'Verses: {len(emb):,}')
print(f'Shape: {vecs.shape}')
print(f'Mean norm: {np.linalg.norm(vecs, axis=1).mean():.4f}')
print(f'Pairwise cosine (sample 10): computing...')
sample = vecs[:10]
norms = np.linalg.norm(sample, axis=1, keepdims=True)
cosines = (sample / norms) @ (sample / norms).T
print(cosines.round(3))
"
```

---

## 8. FAISS Indices — Dual-Space Vector Search

Two separate FAISS indices are built over two different vector spaces. This is
the most important architectural decision in the retrieval layer:

| Index | File | Space | Used for |
|-------|------|-------|----------|
| W2V FAISS | `verse_w2v.faiss` | Word2Vec co-occurrence | User queries, navigator position, semantic search |
| GNN FAISS | `verse_embeddings.faiss` | Graph topology | Oracle: alignment, novelty, next_action |

**Why two spaces?** User text arrives in W2V space (word co-occurrence geometry).
Oracle scoring operates in GNN space (graph topology geometry). They are related
but distinct: the W2V→GNN projection (Step 9) bridges them. Conflating the two
into one index would require projecting every user query before search and lose
the semantic search quality.

Both indices use `IndexFlatIP` (inner product) on L2-normalised vectors,
which is equivalent to cosine similarity search.

### Run
```bash
python index/faiss_store.py
```

Expected output:
```
Building GNN FAISS index over 38927 vectors (dim=32)...
Saved models/verse_embeddings.faiss
Saved models/verse_id_map.pkl
Building W2V FAISS index over 38927 vectors (dim=32)...
Saved models/verse_w2v.faiss
Saved models/verse_w2v_id_map.pkl
```

### Verify
```bash
python -c "
import faiss
gnn_idx = faiss.read_index('models/verse_embeddings.faiss')
w2v_idx = faiss.read_index('models/verse_w2v.faiss')
print(f'GNN FAISS: {gnn_idx.ntotal:,} vectors, dim={gnn_idx.d}')
print(f'W2V FAISS: {w2v_idx.ntotal:,} vectors, dim={w2v_idx.d}')
# Both should print: 38927 vectors, dim=32
"
```

---

## 9. W2V → GNN Projection

### The problem

User text is encoded in W2V space. Oracle functions (alignment, novelty,
next_action) operate in GNN space. These are different vector spaces — the
same verse has different coordinates in each.

### The solution

Ridge regression: fit a linear map from W2V vectors to GNN vectors over all
38,927 verse pairs.

```
W2V_normalised @ coef + intercept ≈ GNN_normalised
coef:      (32, 32)  float32
intercept: (32,)     float32
```

**Why Ridge and not a neural network?** Because the two spaces are nearly
linearly related — the W2V space already captures most of the semantic
structure that the GNN refines. At dim=32, the projection achieves cosine
similarity ≈ 0.993 between projected W2V and actual GNN vectors. A neural
projection would overfit on the training set without meaningfully improving
generalisation.

Alpha (L2 regularisation coefficient) is chosen by 10-fold cross-validation
over the full 38,927 verse pairs, then refitted on all data with the best alpha.

### Run
```bash
python graph/fit_projection.py
```

Expected output:
```
Fitting W2V → GNN projection (Ridge, alpha sweep)...
Best alpha: 0.001  test_cos=0.9930
Refit on all 38927 verses with alpha=0.001
Saved models/w2v_to_gnn_projection.pkl
  coef shape:          (32, 32)
  intercept shape:     (32,)
  mean_projected_cos:  0.9930
```

### Verify
```bash
python -c "
import pickle, numpy as np
proj = pickle.load(open('models/w2v_to_gnn_projection.pkl', 'rb'))
print('coef:', proj['coef'].shape)
print('intercept:', proj['intercept'].shape)
print('mean_projected_cosine:', proj['mean_projected_cosine'])
# expect > 0.98
"
```

---

## 10. Complete Artefacts Checklist

Before starting the server, verify all artefacts exist:

```bash
ls -lh models/
# Required files:
# word2vec.model            (~8 MB)
# verse_vectors.pkl         (~6 MB)
# bible_graph.gpickle       (~5 MB)
# node_map.pkl              (~0.4 MB)
# pyg_data.pt               (~6 MB)
# verse_id_to_idx.pkl       (~0.2 MB)
# gnn_weights.pt            (~12 KB)
# gnn_embeddings.pkl        (~6 MB)
# verse_embeddings.faiss    (~5 MB)
# verse_id_map.pkl          (~0.1 MB)
# verse_w2v.faiss           (~5 MB)
# verse_w2v_id_map.pkl      (~0.1 MB)
# w2v_to_gnn_projection.pkl (~4 KB)
```

Full pipeline summary (run in this order from the project root):
```bash
python data/ingest.py           # Step 3 — corpus
python graph/word2vec.py        # Step 4 — W2V
python graph/builder.py         # Step 5 — graph
python graph/verify_geometry.py # Step 4 validation (optional)
python graph/pyg_data.py        # Step 6 — PyG format
python graph/train.py           # Step 7 — GNN (longest step)
python index/faiss_store.py     # Step 8 — FAISS indices
python graph/fit_projection.py  # Step 9 — projection
```

---

## 11. Launch the API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Expected startup sequence

The FastAPI lifespan hook calls `store.init()` once. Watch for this exact
sequence — each line confirms a component loaded correctly:

```
[store] loaded 38,927 verses  GNN_FAISS=38927  W2V_FAISS=38927
        graph=38927N/59822E  plan_graph=~20000E
[embedder] W2V→GNN projection loaded  mean_cos=0.9930
[oracle] calibrating over 38927 verses → p10=0.2806  p50=0.3327
[typology] 8/8 patterns initialised  (85 verses resolved, 0 skipped)
[typology] seeded program library with 8 canonical typological programs
INFO:     Application startup complete.
```

**Oracle calibration** (`p10=0.2806  p50=0.3327`) is derived entirely from the
model's own reconstruction distribution — it is never hardcoded. The system
scores all 38,927 verses through the GNN's reconstruction head at startup,
then sets:
- `alignment_floor` = 10th percentile of those scores (below = outside prior)
- `alignment_high` = 50th percentile (above = well within manifold)

These thresholds adapt automatically if the model is retrained.

### Verify root endpoint
```bash
curl http://localhost:8000/ | python -m json.tool
```

Expected keys in response: `model`, `corpus`, `verses`, `graph_edges`,
`architecture`, `oracle`, `continual_learning`, `local_models`, `tools`,
`endpoints`.

---

## 12. Verify Each Endpoint

### POST /score — Oracle scoring
```bash
curl -s -X POST http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"content": "He was pierced for our transgressions"}' \
  | python -m json.tool
```
Expect: `alignment` > 0.33 (above `alignment_high` for canonical Scripture),
`typological_pattern.name` = `cross_resurrection`.

### POST /render — Geometric neighbourhood
```bash
curl -s -X POST http://localhost:8000/render \
  -H 'Content-Type: application/json' \
  -d '{"verse_ref": "ISA 53:5"}' \
  | python -m json.tool
```
Expect: nearest verses include Psalm 22, other Servant Song verses,
Romans 5 — cross-canonical resonance the model learned from QUOTES/ECHOES edges.

### POST /simulate — World-state at a canonical period
```bash
curl -s -X POST http://localhost:8000/simulate \
  -H 'Content-Type: application/json' \
  -d '{"books": ["ISA", "JER", "EZK"]}' \
  | python -m json.tool
```
Expect: graph statistics (node count, edge density, betweenness centrality)
for the major prophets subgraph.

### POST /plan — Thematic path between two concepts
```bash
curl -s -X POST http://localhost:8000/plan \
  -H 'Content-Type: application/json' \
  -d '{"from_ref": "GEN 12:1", "to_ref": "HEB 11:8"}' \
  | python -m json.tool
```
Expect: shortest path through the QUOTES+ECHOES subgraph connecting
Abraham's call to Hebrews' faith hall of fame.

### POST /navigate — Full guided search
```bash
curl -s -X POST http://localhost:8000/navigate \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "fixed-interest bond returning 8 percent annually with no equity participation",
    "goal": "covenant-compliant capital structure",
    "max_steps": 3
  }' \
  | python -m json.tool
```
Expect: `truth_layer_summary.dominant_pattern.name` in `covenant` or
`wisdom_two_ways`; `rl_diagnostics.value_function.n_updates` = 3.

### POST /ground — Two-layer truth/factoid response
```bash
curl -s -X POST http://localhost:8000/ground \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Nile watershed structural deterioration — 17% flow reduction",
    "interpret": false
  }' \
  | python -m json.tool
```
Expect: `truth_layer.typological_pattern.name` = `apocalyptic`;
`truth_layer.alfred` contains ARC header and RECOMMENDATION line.

---

## 13. Run the Test Suite

```bash
python tests/test_abada.py
```

Five real Abada capital decision scenarios, each asserting that the Biblical
manifold correctly identifies the structural pattern at stake:

| Scenario | Expected arc |
|----------|-------------|
| Palm oil divestment — supply chain exploitation | `prophetic_indictment` or `wisdom_two_ways` |
| Riba bond — fixed interest, no equity participation | `wisdom_two_ways`, `covenant`, or `prophetic_indictment` |
| East African sovereign compute co-investment | `exodus_deliverance` (high novelty) |
| Third-generation values continuity structure | `exile_restoration` |
| Nile watershed structural deterioration | `apocalyptic` |

Expected: **5/5 assertions pass**.

If any assertion fails, check:
1. `oracle.alignment_floor` and `alignment_high` printed at startup — these
   calibrate automatically and shift if the model was retrained
2. The typological pattern centroids — run `python -c "from services import store; store.init(); from services import typology; print(list(typology._patterns.keys()))"`
3. W2V tokenisation — confirm `graph/word2vec.py` used whitespace split

---

## 14. Architecture — What Was Built and Why

### Two Layers of Reality

```
TRUTH LAYER      Biblical manifold — invariant, eternal, self-consistent
                 Written across 1,500 years / 40+ authors / 3 continents
                 Intertextual graph built from text statistics alone
                 Reconstruction loss = verifiable grounding signal
                         ↓  grounds / interprets
FACTOID LAYER    LLMs + Search + Tools — dynamic, perishable, date-stamped
                 What the world says today
```

The manifold is not another data source. It is the **grounding prior**. Every
factoid output is evaluated against it: does this content fit the manifold?
Which typological arc does it inhabit? What does the eternal pattern say about
the temporal situation?

### Three Oracle Functions from One Frozen Model

The BibleGAT weights are used three ways — all read-only, all from the same
frozen checkpoint:

```python
oracle.next_action(current_vec, candidate_vecs)
# → action probabilities (GNN attention as learned policy)
# Used by: navigator tool selection, typological program search

oracle.alignment_score(content_vec)
# → float [0, 1] (reconstruction likelihood)
# Used by: navigator step reward, /score endpoint, oracle calibration
# The RLVR reward signal — grounded in the manifold, no human labels

oracle.novelty_score(content_vec)
# → float [0, 1] (flipped reconstruction = outside-the-manifold signal)
# Used by: /score endpoint, navigator trajectory summary
```

### Alberta Plan RL Loop

Each navigation step runs all four components simultaneously
(temporal uniformity — no offline training phases):

```
Perception         embedder.encode(content, modality)
                   → 32-dim Biblical coordinate
                   → works for text, image, audio, video, bytes

Reactive Policy    1. program_library.match(current_vec, kind="navigation")
                      → if known sequence matches, execute it directly
                   2. oracle.next_action(current_vec, tool_vecs)
                      → GNN attention weights as action probabilities
                   3. _vf.predict(imagined_next) lookahead
                      → search control when V has signal (n_updates > 5)

Value Function     value_fn.ValueFunction: V(s) = w·s + b
                   33 parameters (32 weights + 1 bias)
                   Updates via TD(λ): γ=0.95, η=0.05, λ=0.8
                   Reward = oracle.alignment_score (manifold-grounded)

Transition Model   _imagine_next(s, action_vec, alpha=0.6)
                   T(s,a) ≈ normalise((1-α)·s + α·embed(tool_description))
                   Frozen — no parameters. Used for k-step lookahead.
```

### SOAR Program Library

After a successful navigation trajectory (mean alignment > 0.65 across all
steps), the trajectory is synthesised into a reusable program:

```python
ProgramEntry(
    name     = "covenant:0001",
    arc      = "covenant",
    trigger  = mean_state_across_trajectory,  # L2-normalised centroid
    sequence = [{"tool": "web_search", "query_template": "..."}, ...],
    reward   = mean_alignment_score,
    kind     = "navigation",  # or "typological" for canonical seeds
)
```

Future queries that land within cosine 0.88 of the trigger bypass oracle
tool selection and execute the stored sequence directly. The library grows
across queries without touching the BibleGAT weights.

### Typological Programs as Canonical Seeds

The eight typological arcs are seeded into the program library at startup
as `kind="typological"` entries. Their sequences contain verse embeddings
(not tool calls):

```python
ProgramEntry(
    kind     = "typological",
    sequence = [{"type": "verse_trajectory", "verse_id": ...,
                 "embedding": 32-dim W2V vector}, ...],
    # for each key verse in the arc's canonical verse list
)
```

Pattern identification runs RL trajectory search:
```
composite(arc) = oracle.next_action(query_vec, arc_centroids)[i]
               × Σ_t γ^t · alignment_score(imagine_next(s_t, verse_t))
```

The arc whose imagined narrative walk from the query position yields highest
cumulative manifold alignment wins. This is program search, not classification.

### Modality-Agnostic Encoding

`embedder.encode(content, modality)` handles any information stream:

```
text    → Word2Vec mean-pool → 32-dim W2V vector
image   → Ollama vision describe → text path
          Fallback: RGB pixel statistics → 32-dim
audio   → Ollama Whisper transcribe → text path
          Fallback: byte statistical features → 32-dim
video   → Sample 8 frames → image path per frame → mean-pool
bytes   → 256-bin byte histogram → (256,32) fixed projection → 32-dim
```

The fixed byte projection is seeded from the W2V vocabulary size — fully
reproducible across restarts.

---

## 15. Extend and Build on Top

### Change embedding dimension

Edit `config.py`:
```python
W2V_VECTOR_SIZE = 64   # was 32
```

Re-run the full pipeline from Step 4. All downstream code adapts — the
dimension flows through `config.W2V_VECTOR_SIZE` to every component.

### Add a typological pattern

Add an entry to `TYPOLOGICAL_PATTERNS` in `services/typology.py`:
```python
"new_arc_name": {
    "description": "...",
    "arc": "Phase 1 → Phase 2 → Phase 3",
    "abada_signal": "Capital interpretation...",
    "key_verses": ["REF 1:1", "REF 2:3", ...],
},
```

Restart the server. `init_typology()` runs automatically and seeds the new
program into the library. No retraining required.

### Add a new modality

Implement `_encode_<modality>(content: Any) -> np.ndarray` in
`services/embedder.py`, add the modality to `_dispatch()`, and update
`_detect_modality()` if auto-detection is needed.

### Add a new tool

1. Create `tools/my_tool.py` inheriting from `tools/base.py`
2. Register in `tools/registry.py`:
   ```python
   "my_tool": lambda: MyTool()
   ```
3. The navigator picks it up automatically on next startup.

### Retrain with a larger corpus

Add a new parser in `data/ingest.py` following the pattern of `_parse_usfx`
or `_parse_enoch_gutenberg`. Run `python data/ingest.py`. The graph builder,
Word2Vec, and GNN all re-run over the expanded corpus.

---

## 16. Open-Source Stack Summary

Every component uses open-source software. No proprietary API. No licence fee.
Reproducible on air-gapped hardware.

| Layer | Library | Version | Role |
|-------|---------|---------|------|
| ML framework | PyTorch | 2.5.1 | GNN training, tensor ops |
| Graph neural network | PyTorch Geometric | 2.6.1 | GATConv, Data object |
| Word embeddings | Gensim | 4.3.3 | Word2Vec training |
| Vector search | FAISS-cpu | 1.9.0 | k-NN over 38k vectors |
| Graph analysis | NetworkX | 3.4.2 | Bible graph, pathfinding |
| Linear projection | scikit-learn | 1.6.0 | Ridge regression W2V→GNN |
| API server | FastAPI + Uvicorn | 0.115.6 | REST endpoints |
| Database | SQLite + SQLAlchemy | built-in + 2.0.36 | Corpus storage |
| LLM inference | Ollama | latest | Local open-source models |
| Web search | DuckDuckGo (httpx) | no key | Factoid retrieval |
| Image encoding | Pillow (PIL) | any | Pixel fallback |

---

## Empirical Results (what was observed)

| Metric | Value | Meaning |
|--------|-------|---------|
| val_perplexity | 1.0837 | GNN reconstructs masked verses near-perfectly |
| W2V→GNN projection cosine | 0.993 | The two spaces are nearly linearly related |
| alignment_floor (p10) | 0.2806 | Below this: content outside the manifold |
| alignment_high (p50) | 0.3327 | Above this: content well within the manifold |
| Typological arc spread (raw cosine) | 0.024 | Before mean-centering: all arcs look identical |
| Typological arc spread (centered) | 0.79–1.63 | After mean-centering: clear discrimination |
| Test suite | 5/5 | All Abada scenarios identify correct arc |
| BibleGAT parameters | 704 | The smallest viable graph attention network |
| Value function parameters | 33 | Updates at every navigation step |

---

*The Bible is the ultimate source because no other text has been written,
copied, translated, cross-referenced, and verified across 1,500 years, 40 authors,
and three continents — and converged. That convergence, computed from raw text
statistics alone, is the prior this system is built on.*
