"""
Verify the 2D Word2Vec geometry before building anything on top of it.

Checks:
  1. Raw scatter — are points separated at all, or collapsed to a point?
  2. Testament separation — do OT/NT/DC/Ethiopian regions differ?
  3. Variance along each axis — is one axis dead (all embeddings flat)?
  4. Nearest-neighbour coherence — do the k closest verses share book/theme?
  5. Inter-testament distance — is NT geometrically distinct from OT?
  6. Token-level spot-checks — where do "God", "LORD", "Jesus", "sin" sit?

Produces:  models/geometry_check.png
           models/geometry_report.txt
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

from config import MODELS_DIR
from db.session import SessionLocal
from db.models import Book, Chapter, Verse

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with open(MODELS_DIR / "verse_vectors.pkl", "rb") as f:
    verse_vectors: dict[int, np.ndarray] = pickle.load(f)

model = Word2Vec.load(str(MODELS_DIR / "word2vec.model"))

db = SessionLocal()
rows = (
    db.query(Verse.id, Book.testament, Book.abbreviation, Book.canonical_order,
             Chapter.number.label("chap"), Verse.number.label("vnum"))
    .join(Chapter, Verse.chapter_id == Chapter.id)
    .join(Book, Chapter.book_id == Book.id)
    .order_by(Book.canonical_order, Chapter.number, Verse.number)
    .all()
)
db.close()

ids        = [r.id        for r in rows]
testaments = [r.testament for r in rows]
books      = [r.abbreviation for r in rows]
book_order = [r.canonical_order for r in rows]

# Align to verse_vectors dict order
vecs = np.stack([verse_vectors[vid] for vid in ids])   # (N, 2)
N = len(vecs)

# ---------------------------------------------------------------------------
# 1. Raw statistics
# ---------------------------------------------------------------------------
report = []
report.append("=== Geometry Verification Report ===\n")
report.append(f"Verses: {N:,}   Dimensions: {vecs.shape[1]}")
D = vecs.shape[1]
for axis in range(D):
    report.append(f"Axis-{axis}  mean={vecs[:,axis].mean():.4f}  std={vecs[:,axis].std():.4f}  "
                  f"range=[{vecs[:,axis].min():.4f}, {vecs[:,axis].max():.4f}]")

# PCA variance breakdown — shows how many axes carry real signal
pca_full = PCA(n_components=D)
pca_full.fit(vecs)
report.append("\nPCA variance explained per axis:")
for i, ratio in enumerate(pca_full.explained_variance_ratio_):
    report.append(f"  PC{i}: {ratio*100:.1f}%")

# Collapse check
for axis in range(D):
    std = vecs[:, axis].std()
    status = "OK" if std > 1e-4 else "COLLAPSED — no signal"
    report.append(f"  axis {axis}: {status}")

report.append("")

# ---------------------------------------------------------------------------
# 2. Per-testament centroid distances
# ---------------------------------------------------------------------------
TCOLORS = {"OT": "#1f77b4", "NT": "#d62728",
           "Deuterocanon": "#2ca02c", "Ethiopian": "#ff7f0e"}

testament_groups: dict[str, np.ndarray] = {}
for t in set(testaments):
    mask = np.array([x == t for x in testaments])
    testament_groups[t] = vecs[mask]

report.append("Per-testament centroids:")
centroids = {}
for t, g in testament_groups.items():
    c = g.mean(axis=0)
    centroids[t] = c
    report.append(f"  {t:15s}  n={len(g):5,}  centroid=({c[0]:.4f}, {c[1]:.4f})  "
                  f"intra-std=({g[:,0].std():.4f}, {g[:,1].std():.4f})")

report.append("\nInter-testament centroid distances:")
ts = list(centroids.keys())
for i in range(len(ts)):
    for j in range(i+1, len(ts)):
        d = np.linalg.norm(centroids[ts[i]] - centroids[ts[j]])
        report.append(f"  {ts[i]:15s} ↔ {ts[j]:15s}  {d:.4f}")

report.append("")

# ---------------------------------------------------------------------------
# 3. Nearest-neighbour coherence (same-book rate in top-5)
# ---------------------------------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity

# Sample 200 random verses for NN check (full N×N is expensive at N=38k)
rng = np.random.default_rng(0)
sample_idx = rng.choice(N, size=min(200, N), replace=False)
sample_vecs = vecs[sample_idx]
sample_books = [books[i] for i in sample_idx]

sim_matrix = cosine_similarity(sample_vecs)
np.fill_diagonal(sim_matrix, -1)   # exclude self

same_book_hits = 0
total_checked  = 0
for i in range(len(sample_idx)):
    top5 = np.argsort(sim_matrix[i])[-5:]
    for j in top5:
        if sample_books[j] == sample_books[i]:
            same_book_hits += 1
        total_checked += 1

coherence_pct = 100 * same_book_hits / total_checked
report.append(f"NN coherence (same-book rate in top-5, 200-verse sample): "
              f"{coherence_pct:.1f}%")
report.append("  (random baseline ≈ 1–5% depending on book length)")
report.append("")

# ---------------------------------------------------------------------------
# 4. Token-level spot-checks
# ---------------------------------------------------------------------------
probes = ["God", "LORD", "Jesus", "Israel", "sin", "covenant",
          "YHWH", "Enoch", "angel", "righteousness"]
report.append("Token nearest-neighbours (top 3):")
for word in probes:
    if word in model.wv:
        nn = model.wv.most_similar(word, topn=3)
        report.append(f"  {word:15s} → {[w for w,_ in nn]}")
    else:
        report.append(f"  {word:15s} → NOT IN VOCAB")

report.append("")

# ---------------------------------------------------------------------------
# 5. Book-level canonical spread — use PC0 (highest-variance axis)
# ---------------------------------------------------------------------------
pca2 = PCA(n_components=2)
vecs_2d = pca2.fit_transform(vecs)   # project to 2D for all plots

by_book_order: dict[int, list[float]] = {}
for i, bo in enumerate(book_order):
    by_book_order.setdefault(bo, []).append(vecs_2d[i, 0])

book_order_vals = sorted(by_book_order.keys())
book_means_pc0 = [np.mean(by_book_order[bo]) for bo in book_order_vals]
correlation = np.corrcoef(book_order_vals, book_means_pc0)[0, 1]
report.append(f"\nCorrelation of canonical order with PC0: {correlation:.4f}")
report.append("  (+1 = early books left, late books right; 0 = no separation)")
report.append(f"PCA2 variance explained: PC0={pca2.explained_variance_ratio_[0]*100:.1f}%  "
              f"PC1={pca2.explained_variance_ratio_[1]*100:.1f}%")

# Per-testament 2D centroids (PCA projected)
testament_groups_2d: dict[str, np.ndarray] = {}
for t in set(testaments):
    mask = np.array([x == t for x in testaments])
    testament_groups_2d[t] = vecs_2d[mask]

from config import W2V_VECTOR_SIZE, W2V_WINDOW, W2V_EPOCHS

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Verse Embeddings — Biblical Corpus  "
    f"(dim={W2V_VECTOR_SIZE}, window={W2V_WINDOW}, epochs={W2V_EPOCHS})  "
    f"— PCA projection to 2D",
    fontsize=11,
)

# Left: by testament
ax = axes[0]
for t, g in testament_groups_2d.items():
    ax.scatter(g[:, 0], g[:, 1], s=2, alpha=0.4,
               color=TCOLORS.get(t, "gray"), label=t, rasterized=True)
ax.set_title("By testament")
ax.set_xlabel("PC0")
ax.set_ylabel("PC1")
ax.legend(markerscale=4, fontsize=8)

# Right: by canonical order
ax = axes[1]
sc = ax.scatter(vecs_2d[:, 0], vecs_2d[:, 1], s=1, alpha=0.3,
                c=book_order, cmap="plasma", rasterized=True)
plt.colorbar(sc, ax=ax, label="canonical book order")
ax.set_title("By canonical order")
ax.set_xlabel("PC0")
ax.set_ylabel("PC1")

plt.tight_layout()
out_png = MODELS_DIR / "geometry_check.png"
plt.savefig(out_png, dpi=150)
print(f"Plot saved → {out_png}")

out_txt = MODELS_DIR / "geometry_report.txt"
report_text = "\n".join(report)
out_txt.write_text(report_text)
print(f"Report saved → {out_txt}")
print()
print(report_text)
