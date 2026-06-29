"""
Modal encoder — maps any stream of bits/bytes into the 8-dim Biblical
coordinate space learned by the GNN.

The Biblical manifold is the universal coordinate system.  Every modality
(text, image, audio, video, binary) is projected onto it so the world model
geometry can guide navigation regardless of source modality.

Projection strategy per modality
─────────────────────────────────
text         Word2Vec mean-pool → soft k-NN projection onto GNN manifold
             (same tokenisation as training: whitespace-split)

image        If a vision model is available via Ollama → describe → text path
             Otherwise: pixel statistics (mean, std per channel + entropy) → 8-dim

audio        If Whisper/Ollama audio model available → transcribe → text path
             Otherwise: byte statistical features → 8-dim

video        Frame-sample → image path per frame → mean pool

bytes/any    Try UTF-8 decode → text path
             Fallback: byte histogram + entropy features → fixed linear map → 8-dim

The fixed random projection (for non-text fallback) is seeded from the
trained Word2Vec model's vocabulary size so it is reproducible across
server restarts and consistent with the learned space.
"""

import hashlib
import struct
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Module-level state — set by _init() via services/store.py
# ---------------------------------------------------------------------------
_w2v_model = None          # gensim Word2Vec
_faiss_index = None        # built on GNN embeddings
_verse_id_list = None      # FAISS row → verse_id
_gnn_embeddings = None     # dict {verse_id: np.ndarray} — raw GNN outputs
_byte_projection: np.ndarray | None = None   # fixed (256, 8) for byte fallback

# Learned W2V → GNN linear projection (fitted by graph/fit_projection.py)
_proj_coef: np.ndarray | None = None         # (8, 8)
_proj_intercept: np.ndarray | None = None    # (8,)


def _init(w2v, faiss_index, verse_id_list, gnn_embeddings) -> None:
    global _w2v_model, _faiss_index, _verse_id_list, _gnn_embeddings
    global _byte_projection, _proj_coef, _proj_intercept

    _w2v_model      = w2v
    _faiss_index    = faiss_index
    _verse_id_list  = verse_id_list
    _gnn_embeddings = gnn_embeddings

    # Load the learned W2V→GNN linear projection
    import pickle
    from config import MODELS_DIR
    proj_path = MODELS_DIR / "w2v_to_gnn_projection.pkl"
    if proj_path.exists():
        with open(proj_path, "rb") as f:
            proj = pickle.load(f)
        _proj_coef       = proj["coef"]        # (8, 8)
        _proj_intercept  = proj["intercept"]   # (8,)
        print(f"[embedder] W2V→GNN projection loaded  "
              f"mean_cos={proj.get('mean_projected_cosine', '?'):.4f}")
    else:
        print("[embedder] WARNING: projection not found — run graph/fit_projection.py")

    # Seeded byte-fallback projection: reproducible across restarts
    seed = len(w2v.wv.key_to_index)
    rng  = np.random.default_rng(seed)
    _byte_projection = rng.standard_normal((256, 8)).astype(np.float32)
    _byte_projection /= np.linalg.norm(_byte_projection, axis=0, keepdims=True) + 1e-9


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def encode(content: Any, modality: str = "auto") -> np.ndarray:
    """
    Map any content into the 8-dim Biblical coordinate space.

    Returns a L2-normalised float32 vector (8,).
    """
    if modality == "auto":
        modality = _detect_modality(content)

    vec = _dispatch(content, modality)
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def encode_bytes(raw: bytes, hint_modality: str = "auto") -> np.ndarray:
    """Encode raw bytes (any source) into W2V Biblical coordinates."""
    if hint_modality == "auto":
        hint_modality = _detect_modality_from_bytes(raw)
    if hint_modality == "text":
        try:
            text = raw.decode("utf-8", errors="replace")
            return encode(text, "text")
        except Exception:
            pass
    return encode(raw, hint_modality)


def encode_gnn(content, modality: str = "auto") -> np.ndarray:
    """
    Encode content into GNN Biblical coordinate space (structural space).

    Pipeline: encode() → W2V vec → linear projection → GNN-space L2-norm.

    Use this only when GNN-space coordinates are explicitly needed (e.g. for
    feeding directly into oracle internals without going through oracle.py).
    For all standard use (search, navigation), use encode() which returns W2V.
    """
    w2v_vec = encode(content, modality)
    if _proj_coef is not None:
        projected = w2v_vec @ _proj_coef + _proj_intercept
        norm = np.linalg.norm(projected)
        return projected / (norm + 1e-9)
    return w2v_vec   # fallback if projection not available


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------

def _detect_modality(content: Any) -> str:
    if isinstance(content, str):   return "text"
    if isinstance(content, bytes): return _detect_modality_from_bytes(content)
    if isinstance(content, list):  return "text"   # list of strings assumed
    return "bytes"


def _detect_modality_from_bytes(raw: bytes) -> str:
    """Sniff bytes for magic numbers."""
    if len(raw) < 4:
        return "bytes"
    sig = raw[:12]
    # JPEG
    if sig[:2] == b'\xff\xd8':                       return "image"
    # PNG
    if sig[:8] == b'\x89PNG\r\n\x1a\n':              return "image"
    # GIF
    if sig[:6] in (b'GIF87a', b'GIF89a'):            return "image"
    # WebP
    if sig[:4] == b'RIFF' and raw[8:12] == b'WEBP':  return "image"
    # MP3
    if sig[:3] == b'ID3' or sig[:2] == b'\xff\xfb':  return "audio"
    # WAV
    if sig[:4] == b'RIFF' and raw[8:12] == b'WAVE':  return "audio"
    # FLAC
    if sig[:4] == b'fLaC':                            return "audio"
    # MP4 / MOV
    if raw[4:8] in (b'ftyp', b'moov'):               return "video"
    # Try UTF-8 text
    try:
        raw[:512].decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        pass
    return "bytes"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _dispatch(content: Any, modality: str) -> np.ndarray:
    if modality == "text":
        return _encode_text(content if isinstance(content, str) else str(content))
    if modality == "image":
        return _encode_image(content)
    if modality == "audio":
        return _encode_audio(content)
    if modality == "video":
        return _encode_video(content)
    # bytes / unknown
    return _encode_bytes_statistical(
        content if isinstance(content, bytes) else str(content).encode()
    )


# ---------------------------------------------------------------------------
# Per-modality encoders
# ---------------------------------------------------------------------------

def _encode_text(text: str) -> np.ndarray:
    """
    Word2Vec mean-pool → L2-normalised W2V vector (semantic space).

    W2V space encodes co-occurrence semantics directly from the Biblical corpus.
    This is the primary encoding for user-facing semantic search and navigator
    position tracking.  Oracle functions project to GNN space internally.
    """
    if _w2v_model is None:
        raise RuntimeError("Embedder not initialised — call store.init() first.")

    tokens = text.split()
    vecs   = [_w2v_model.wv[t] for t in tokens if t in _w2v_model.wv]

    if not vecs:
        return _encode_bytes_statistical(text.encode("utf-8", errors="replace"))

    w2v_vec = np.mean(vecs, axis=0).astype(np.float32)
    return w2v_vec / (np.linalg.norm(w2v_vec) + 1e-9)


def _encode_image(content: Any) -> np.ndarray:
    """
    Image → Biblical coordinates.

    Path 1 (preferred): describe via Ollama vision model → text → _encode_text
    Path 2 (fallback):  pixel statistics (RGB mean/std/entropy per channel) → 8-dim
    """
    raw: bytes = content if isinstance(content, bytes) else bytes(content)

    # Path 1: vision model description
    try:
        from tools.registry import get as get_tool
        vision_tool = get_tool("ollama_vision")
        result = vision_tool.invoke(
            "Describe this image concisely in 2-3 sentences, focusing on the "
            "main subject, context, and any text visible.",
            image_bytes=raw,
        )
        return _encode_text(result.content)
    except (KeyError, RuntimeError):
        pass

    # Path 2: pixel statistics
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((64, 64))
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (64,64,3)
        features = np.array([
            arr[:, :, 0].mean(), arr[:, :, 0].std(),
            arr[:, :, 1].mean(), arr[:, :, 1].std(),
            arr[:, :, 2].mean(), arr[:, :, 2].std(),
            float(-(arr + 1e-9) * np.log(arr + 1e-9) - (1 - arr + 1e-9) * np.log(1 - arr + 1e-9)).mean(),
            float(arr.mean()),
        ], dtype=np.float32)
        return features
    except Exception:
        return _encode_bytes_statistical(raw)


def _encode_audio(content: Any) -> np.ndarray:
    """
    Audio → Biblical coordinates.

    Path 1: transcribe via Whisper (if available locally) → text path
    Path 2: byte statistical features
    """
    raw: bytes = content if isinstance(content, bytes) else bytes(content)

    # Path 1: Whisper transcription via Ollama (if supported)
    try:
        import ollama
        # whisper model in Ollama (not yet widely supported; attempt anyway)
        resp = ollama.generate(model="whisper", prompt="", images=[raw])
        if resp.get("response"):
            return _encode_text(resp["response"])
    except Exception:
        pass

    return _encode_bytes_statistical(raw)


def _encode_video(content: Any) -> np.ndarray:
    """
    Video → Biblical coordinates via frame sampling.

    Samples up to 8 evenly-spaced frames → encode each as image → mean pool.
    Falls back to byte statistics if cv2/PIL not available.
    """
    raw: bytes = content if isinstance(content, bytes) else bytes(content)

    try:
        import cv2
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(raw); tmpfile = f.name
        cap = cv2.VideoCapture(tmpfile)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_vecs = []
        for i in range(min(8, total)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / 8))
            ok, frame = cap.read()
            if not ok: continue
            _, buf = cv2.imencode(".jpg", frame)
            frame_vecs.append(_encode_image(buf.tobytes()))
        cap.release()
        os.unlink(tmpfile)
        if frame_vecs:
            return np.mean(frame_vecs, axis=0).astype(np.float32)
    except Exception:
        pass

    return _encode_bytes_statistical(raw)


def _encode_bytes_statistical(raw: bytes) -> np.ndarray:
    """
    Universal fallback: byte histogram + entropy features → fixed linear map → 8-dim.

    The 256-bin byte histogram is L1-normalised (forms a probability distribution).
    It is multiplied by the fixed (256, 8) projection matrix seeded from the
    W2V vocab size — fully reproducible across restarts.
    """
    if not raw:
        return np.zeros(8, dtype=np.float32)

    # 256-bin byte histogram
    hist = np.zeros(256, dtype=np.float32)
    for b in raw[:65536]:   # cap at 64KB for speed
        hist[b] += 1.0
    hist /= (hist.sum() + 1e-9)   # L1 normalise

    if _byte_projection is not None:
        vec = hist @ _byte_projection   # (8,)
    else:
        vec = hist[:8].copy()   # emergency fallback before _init

    return vec.astype(np.float32)
