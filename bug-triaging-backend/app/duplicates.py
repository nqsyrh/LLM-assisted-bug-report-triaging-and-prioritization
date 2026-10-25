"""
duplicates.py
Live duplicate detection for newly-submitted reports, against the existing
499-report corpus. Same model (all-MiniLM-L6-v2), same similarity metric
(cosine via normalized inner product), same threshold (0.80) as
find_duplicates.py.

Strategy: the FAISS index over the existing 499 reports is built once at
startup (or on first use) from whatever's currently in the database, then
reused for every new-report submission. Rebuilding from scratch per request
would be wasteful and isn't how your batch script works either — it embeds
the whole corpus once.
"""
import numpy as np
from sqlalchemy.orm import Session

from . import models

SIMILARITY_THRESHOLD = 0.80  # identical to find_duplicates.py

_embedder = None
_index = None
_index_ids: list[int] = []  # maps FAISS row position -> bug report id


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def build_index(db: Session):
    """
    (Re)builds the FAISS index from all bug_reports currently in the DB.
    Called once at seed time, and can be re-called if you want the index
    to reflect newly-submitted reports too.
    """
    global _index, _index_ids

    reports = db.query(models.BugReport).all()
    if not reports:
        _index = None
        _index_ids = []
        return

    import faiss

    texts = [r.full_text or "" for r in reports]
    ids = [r.id for r in reports]

    embedder = get_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    _index = index
    _index_ids = ids


def find_duplicate_for_text(text: str, db: Session, k: int = 3):
    """
    For a single NEW report's text, finds the closest existing report.
    Returns (is_duplicate, matched_id, similarity_score) — matched_id and
    similarity_score are None if nothing crosses the threshold.

    Same logic as find_duplicates.py's neighbor search, just for one
    incoming item against the prebuilt index instead of all-pairs.
    """
    global _index, _index_ids

    if _index is None or _index.ntotal == 0:
        build_index(db)
        if _index is None or _index.ntotal == 0:
            return False, None, None

    import faiss

    embedder = get_embedder()
    query_vec = embedder.encode([text or ""])
    query_vec = np.array(query_vec).astype("float32")
    faiss.normalize_L2(query_vec)

    k_search = min(k, _index.ntotal)
    similarities, indices = _index.search(query_vec, k_search)

    best_similarity = float(similarities[0][0])
    best_idx = indices[0][0]

    if best_similarity >= SIMILARITY_THRESHOLD:
        matched_id = _index_ids[best_idx]
        return True, matched_id, round(best_similarity, 3)

    return False, None, None
