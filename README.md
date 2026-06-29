# Biblical World Model

A 32-dimensional geometric prior trained on 38,927 Ethiopian Orthodox Bible verses — the grounding truth layer for Abada Group's **BRAIN** faith-directed capital advisory system.

---

## This is not a Bible search engine

It is a **universal coordinate system**. Any stream of bits/bytes — text, image, audio, video, binary data, foundation model outputs — is projected onto a 32-dimensional Biblical manifold. The geometry of that manifold then guides which tools to invoke next and how to connect what's been retrieved.

```
Any bits/bytes (any modality)
      ↓
Modal Encoder (text→Word2Vec, image→Ollama vision, audio→Whisper, bytes→statistical)
      ↓
32-dim Biblical Coordinate Space  ← the invariant
      ↓
World Model Geometry (render / simulate / plan)
      ↓
Tool Guidance (what to call, with what context)
      ↓
Tool Layer: Foundation models (Claude / GPT-4o / Gemini) + web search + fetch + APIs
      ↓
Tool outputs → encode → update position → plan next step → repeat
```

The loop continues until the oracle declares convergence — when the trajectory's alignment with the Biblical manifold is high enough to synthesise a program.

---

## Architecture

A **BibleGAT** — a 704-parameter Graph Attention Network — is trained self-supervised on the Biblical graph and then **frozen**. The frozen weights are reused three orthogonal ways simultaneously, with no additional training:

| Use | Mechanism | What it answers |
|-----|-----------|----------------|
| **Next action** | Attention weights = action probabilities | Which concept / node to attend to next |
| **Alignment / truth scoring** | Reconstruction likelihood | How well does the Biblical manifold explain this content? |
| **Novelty detection** | Flipped reconstruction logit | Is this genuinely outside the known prior? |

This triple unification is grounded in the result that [Transformers are Graph Neural Networks](https://arxiv.org/pdf/2506.22084) — BibleGAT IS a transformer operating on the Biblical graph, so the same attention and reconstruction heads that worked during training are valid inference mechanisms at runtime.

Above the oracle, a **TD(λ) value function** (33 parameters: 32 weights + 1 bias) updates via the Alberta Plan RL loop. Biblical alignment scores serve as grounded rewards — no human labels required.

Two FAISS indices run in parallel:
- **W2V semantic space** — user-facing search (Word2Vec, 32-dim)
- **GNN structural space** — oracle internals (BibleGAT embeddings, 32-dim)

A ridge-regression projection (mean cosine similarity = 0.993) bridges the two.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete specification.

---

## Corpus

| Source | Books | Verses |
|--------|-------|--------|
| World English Bible (WEB) | 66 | ~31,000 |
| WEB Deuterocanon | 15 | ~5,700 |
| 1 Enoch (Project Gutenberg) | 1 | ~1,000 |
| **Total** | **82** | **38,927** |

Graph: 38,927 nodes, ~59,822 edges across three edge types:
- **PRECEDES** — canonical sequential order
- **QUOTES** — 4-char n-gram overlap ≥ 0.6 across different books (~14k edges)
- **ECHOES** — TF-IDF cosine ≥ 0.45, minimum 1 book apart (~6k edges)

---

## Quick start (artifacts committed — no retraining needed)

```bash
# 1. Clone and install
git clone https://github.com/joshuwaifo/biblical-world-model
cd biblical-world-model
pip install -r requirements.txt

# 2. Install Ollama for vision/LLM tool layer (https://ollama.com)
ollama pull llava

# 3. Start the server
uvicorn routes.main:app --reload

# 4. Score any content against the Biblical prior
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"content": "fixed-interest bond with no equity participation", "modality": "text"}' \
  | python -m json.tool

# 5. Run a guided navigation
curl -s -X POST http://localhost:8000/navigate \
  -H "Content-Type: application/json" \
  -d '{"query": "riba — interest-bearing lending", "goal": "Quranic and Biblical convergence on usury", "max_steps": 5}' \
  | python -m json.tool
```

---

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/score` | POST | Score any content (text, image, audio, video, bytes) against the oracle. Returns alignment, novelty, Biblical coordinates, nearest verses. |
| `/navigate` | POST | Guided search with two-layer response: truth layer (manifold interpretation) + factoid layer (sources, tools, snippets). Runs the full RL loop. |
| `/render` | POST | Geometric k-NN neighbourhood of a verse reference or concept. |
| `/plan` | POST | Shortest thematic path between two concepts via QUOTES+ECHOES subgraph. |
| `/simulate` | POST | World-state statistics at canonical periods (node count, edge density, betweenness centrality). |

Full request/response schemas: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Full reproduction cycle

All trained artifacts are committed — you can go straight to Step 10. This section documents the full pipeline if you want to retrain from scratch.

**Requirements:** Python 3.10+, 8+ GB RAM, multi-core CPU. No GPU required. No proprietary APIs.

```
Step 1  — Clone and install
          git clone https://github.com/joshuwaifo/biblical-world-model
          pip install -r requirements.txt
          ollama pull llava          # for vision modality

Step 2  — Ingest the Biblical corpus  (~15 min)
          python data/ingest.py
          # Downloads: WEB Bible (USFX), Deuterocanon, 1 Enoch
          # Produces:  bible.db  with 38,927 verses
          # SKIP if using the committed bible.db

Step 3  — Train Word2Vec embeddings  (~2 min)
          python graph/word2vec.py
          # Produces:  models/word2vec.model
          #            models/verse_vectors.pkl

Step 4  — Build the intertextual graph  (~5 min)
          python graph/builder.py
          # Produces:  models/bible_graph.gpickle  (~59,822 edges)

Step 5  — Convert to PyTorch Geometric format  (~1 min)
          python graph/pyg_data.py
          # Produces:  models/pyg_data.pt

Step 6  — Train BibleGAT  (~10 min, CPU only)
          python graph/train.py
          # Produces:  models/gnn_weights.pt  (frozen from here on)
          #            models/gnn_embeddings.pkl
          # Target:    val_perplexity ≤ 1.09  (achieved: 1.0837)

Step 7  — Fit W2V→GNN projection  (<1 min)
          python graph/fit_projection.py
          # Produces:  models/w2v_to_gnn_projection.pkl
          # Target:    mean cosine ≥ 0.99  (achieved: 0.993)

Step 8  — Build FAISS indices  (~1 min)
          python index/faiss_store.py
          # Produces:  models/verse_w2v.faiss  +  models/verse_w2v_id_map.pkl
          #            models/verse_embeddings.faiss  +  models/verse_id_map.pkl

Step 9  — Verify geometry  (optional, recommended)
          python graph/verify_geometry.py
          # Produces:  models/geometry_report.txt
          #            models/geometry_check.png

Step 10 — Launch the server
          uvicorn routes.main:app --reload
          # Server ready at http://localhost:8000
```

**If artifacts are committed (default):** skip Steps 2–9. Run Step 10 immediately.

---

## Trained model artifacts

| File | Size | Description |
|------|------|-------------|
| `models/gnn_weights.pt` | 12 KB | BibleGAT trained weights (frozen at runtime) |
| `models/word2vec.model` | 8 MB | Gensim Word2Vec (32-dim, 28,712 vocab) |
| `models/verse_vectors.pkl` | 6 MB | verse_id → 32-dim W2V vector |
| `models/gnn_embeddings.pkl` | 6 MB | verse_id → 32-dim GNN embedding |
| `models/pyg_data.pt` | 6 MB | PyTorch Geometric Data object |
| `models/bible_graph.gpickle` | 5 MB | NetworkX DiGraph (38,927 nodes, ~59,822 edges) |
| `models/verse_w2v.faiss` | 5 MB | W2V-space FAISS index (semantic search) |
| `models/verse_embeddings.faiss` | 5 MB | GNN-space FAISS index (oracle internals) |
| `models/w2v_to_gnn_projection.pkl` | 4 KB | Ridge regression bridge (32×32 coef + intercept) |
| `models/geometry_check.png` | 740 KB | 2D PCA coloured by testament |
| `models/geometry_report.txt` | 4 KB | Testament separation, axis variance, NN coherence |

**Empirical results:** val_perplexity 1.0837 · W2V→GNN cosine 0.993 · 5/5 test scenarios pass · 704 BibleGAT parameters · 33 value function parameters (updates every step)

---

## Test suite

Five integration scenarios for Abada Group capital advisory — all pass at dim=32:

```bash
pytest tests/test_abada.py -v
```

| Scenario | Expected arc |
|----------|-------------|
| Riba screen (fixed-interest bond) | `covenant` or `wisdom_two_ways` |
| Divestment (palm oil exploitation) | `prophetic_indictment` |
| Co-investment (East African sovereign compute) | `exodus_deliverance` |
| Family governance (third-generation values) | `exile_restoration` |
| Bioregional risk (Nile watershed) | `apocalyptic` |

---

## Hackathon submission context

The system is being submitted under two primary claims:

**Continual Learning (primary)** — the TD(λ) value function updates on every navigation step. The reward signal comes from the Biblical manifold itself (oracle alignment score), not human labels. Zero user intervention required after deployment.

**Recursive Intelligence (secondary)** — BibleGAT's own attention weights guide the search for programs that improve future navigation. The model uses its own emergent geometry to bootstrap its own improvement loop (SOAR-style program library, pruned by use-count).

**Judging criteria:**
- Technicality 40% — custom GNN training, dual-space FAISS, Alberta Plan RL loop, self-calibrating oracle
- Creativity 25% — the AlphaProof analogy, pattern-matching as program search, modality-agnostic truth grounding
- Live Demo 20% — exact demo script with terminal commands and pre-written Q&A in ARCHITECTURE.md §27e
- Future Potential 15% — non-gradient continual learning, grounded reward without RLHF, generalises to any structured corpus

**Sponsor angles:** DigitalOcean (infrastructure), LiveKit (the embedder already accepts raw audio bytes natively — no preprocessing needed).

---

## Reading & watching list

Intellectual lineage of the project — recommended reading/watching before diving into the code:

| Resource | Author | Why it matters |
|----------|--------|---------------|
| [Emergent Properties in Neural Networks](https://youtu.be/E22AOHAEtu4?si=YJfZlHbhi6FVeSzP) | Shuchao Bi | Foundation for understanding how structure emerges from scale in geometric learning |
| [A Functional Taxonomy of World Models](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models) | Fei-Fei Li | Situates this project within the broader landscape of world model architectures |
| [Transformers are Graph Neural Networks](https://arxiv.org/pdf/2506.22084) | Chaitanya Joshi (Cambridge) | Proves BibleGAT IS a transformer — the key insight behind using the same weights for three orthogonal purposes simultaneously |
| [What AI Reveals About the Universe](https://howiwrite.substack.com/p/john-lennox-what-ai-reveals-about) | John Lennox (Oxford Mathematician) | The Word-based structure of reality — theological foundation for grounding an AI prior in Scripture |
| [The Bible as the World's First Hyperlinked Text](https://philosophadam.wordpress.com/2018/05/16/the-first-hyperlinked-text-the-bible-and-its-63779-cross-references/) | Dr. Jordan Peterson | 63,779 cross-references — the intellectual basis for QUOTES and ECHOES graph edges |
| [OaK Architecture — NeurIPS 2025](https://neurips.cc/virtual/2025/loc/mexico-city/invited-talk/129132) | Richard Sutton | World modelling + reactive policies + options + generalisable value functions — direct ancestor of the RL loop in `services/rl_loop.py` |
| [Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) | David Silver (DeepMind) | Generalisable reward functions + infinite information streams + RL — maps directly to how Biblical alignment scores serve as grounded rewards in this system |

---

## Technical stack

Python 3.10+ · PyTorch 2.5.1 · PyTorch Geometric 2.6.1 · Gensim 4.3.3 · FAISS-cpu 1.9.0 · NetworkX 3.4.2 · scikit-learn 1.6.0 · FastAPI 0.115.6 · SQLite · Ollama (local LLM inference, no cloud dependencies)

---

## License

MIT
