"""
Train Word2Vec on the Biblical corpus.

Starting from the most irreducible configuration:
  - 2 dimensions  (minimum for any geometry; you can visualise directly)
  - window = 1    (immediate neighbours only — most local context signal)
  - min_count = 1 (keep all words; the corpus is small enough)
  - 1 epoch       (single pass; see if co-occurrence signal is already present)

Run:  python graph/word2vec.py
Expand dims geometrically (2→4→8→16→32) only when 2D clusters are saturated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from gensim.models import Word2Vec

from config import MODELS_DIR, W2V_EPOCHS, W2V_MIN_COUNT, W2V_VECTOR_SIZE, W2V_WINDOW
from db.session import SessionLocal
from db.models import Verse

W2V_PATH = MODELS_DIR / "word2vec.model"
VERSE_VECTORS_PATH = MODELS_DIR / "verse_vectors.pkl"   # {verse_id: np.array}

def tokenise(text: str) -> list[str]:
    # Whitespace-split only: the minimal, zero-assumption boundary.
    # "God," "God." "God" remain distinct tokens — punctuation, case,
    # transliterations (YHWH, Elohim), numbers, and every surface-form
    # distinction the scribes made are all preserved.
    return text.split()


def load_corpus() -> tuple[list[list[str]], list[int]]:
    db = SessionLocal()
    verses = db.query(Verse).order_by(Verse.id).all()
    sentences = [tokenise(v.text) for v in verses]
    ids       = [v.id for v in verses]
    db.close()
    return sentences, ids


def train():
    print(f"Loading corpus...")
    sentences, verse_ids = load_corpus()
    print(f"  {len(sentences):,} verses, vocabulary will be built from scratch")

    print(f"Training Word2Vec  dim={W2V_VECTOR_SIZE}  window={W2V_WINDOW}  "
          f"min_count={W2V_MIN_COUNT}  epochs={W2V_EPOCHS}")
    model = Word2Vec(
        sentences=sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=1,           # skip-gram (predicts context from target word)
        epochs=W2V_EPOCHS,
        workers=4,
        seed=0,
    )
    model.save(str(W2V_PATH))
    print(f"  saved → {W2V_PATH}")
    print(f"  vocab size: {len(model.wv):,} tokens")

    # Build verse embeddings: mean-pool token vectors
    verse_vectors: dict[int, np.ndarray] = {}
    for verse_id, tokens in zip(verse_ids, sentences):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if vecs:
            verse_vectors[verse_id] = np.mean(vecs, axis=0).astype(np.float32)
        else:
            verse_vectors[verse_id] = np.zeros(W2V_VECTOR_SIZE, dtype=np.float32)

    with open(VERSE_VECTORS_PATH, "wb") as f:
        pickle.dump(verse_vectors, f)
    print(f"  verse vectors saved → {VERSE_VECTORS_PATH}")

    # Sanity check: nearest neighbours of "God" (case preserved).
    # At dim=2, epoch=1 results will be noisy — that is expected.
    # Expand: epochs 1→5→20→50; dim 2→4→8→16→32 once geometry saturates.
    for probe in ("God", "LORD", "Jesus", "Israel"):
        if probe in model.wv:
            similar = model.wv.most_similar(probe, topn=3)
            print(f"  nearest to '{probe}': {[w for w,_ in similar]}")

    return model, verse_vectors


if __name__ == "__main__":
    train()
