"""Composite similarity for poem-song pairs using embeddings + features."""

from typing import Dict, Optional

import numpy as np


def compute_composite_similarity(
    poem_embedding: np.ndarray,
    song_embedding: np.ndarray,
    poem_features: Dict[str, np.ndarray],
    song_features: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute weighted similarity using embeddings + semantic/structural/lexical features.

    Args:
        poem_embedding: (768,) embedding vector
        song_embedding: (768,) embedding vector
        poem_features: dict with keys {structural, semantic, lexical}
        song_features: dict with keys {structural, semantic, lexical}
        weights: optional weights dict {cosine, semantic, structural, lexical}
                 (defaults: 0.4, 0.35, 0.15, 0.1)

    Returns:
        dict: {composite, cosine, semantic, structural, lexical} all in [0, 1]

    Notes:
        Both poems and songs have 3 structural features (line_count, rhyme_density, avg_line_length).
        All similarities use cosine distance, normalized to [0, 1].
    """
    if weights is None:
        weights = {"cosine": 0.4, "semantic": 0.35, "structural": 0.15, "lexical": 0.1}

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        weights = {"cosine": 0.25, "semantic": 0.25, "structural": 0.25, "lexical": 0.25}

    # Cosine similarity (embeddings)
    poem_norm = np.linalg.norm(poem_embedding)
    song_norm = np.linalg.norm(song_embedding)
    if poem_norm > 0 and song_norm > 0:
        cosine_sim = np.dot(poem_embedding, song_embedding) / (poem_norm * song_norm)
        cosine_sim = (cosine_sim + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    else:
        cosine_sim = 0.0

    # Semantic similarity (42-dim)
    p_sem = poem_features["semantic"]
    s_sem = song_features["semantic"]
    p_norm = np.linalg.norm(p_sem)
    s_norm = np.linalg.norm(s_sem)
    semantic_sim = np.dot(p_sem, s_sem) / (p_norm * s_norm) if p_norm > 0 and s_norm > 0 else 0.0
    semantic_sim = (semantic_sim + 1.0) / 2.0

    # Structural similarity (first 3 features)
    p_struct = poem_features["structural"][:3]
    s_struct = song_features["structural"][:3]
    p_norm = np.linalg.norm(p_struct)
    s_norm = np.linalg.norm(s_struct)
    structural_sim = np.dot(p_struct, s_struct) / (p_norm * s_norm) if p_norm > 0 and s_norm > 0 else 0.0
    structural_sim = (structural_sim + 1.0) / 2.0

    # Lexical similarity (3-dim)
    p_lex = poem_features["lexical"]
    s_lex = song_features["lexical"]
    p_norm = np.linalg.norm(p_lex)
    s_norm = np.linalg.norm(s_lex)
    lexical_sim = np.dot(p_lex, s_lex) / (p_norm * s_norm) if p_norm > 0 and s_norm > 0 else 0.0
    lexical_sim = (lexical_sim + 1.0) / 2.0

    composite = (
        weights["cosine"] * cosine_sim +
        weights["semantic"] * semantic_sim +
        weights["structural"] * structural_sim +
        weights["lexical"] * lexical_sim
    )

    return {
        "composite": float(composite),
        "cosine": float(cosine_sim),
        "semantic": float(semantic_sim),
        "structural": float(structural_sim),
        "lexical": float(lexical_sim),
    }
