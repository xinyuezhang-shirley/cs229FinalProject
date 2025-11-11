"""Retrieve nearest songs for each poem in the aligned embedding space.

This script reuses the alignment utilities from `modality_alignment.py` to
place song and poem MPNet embeddings into the shared space (per-modality
z-score → modality-direction removal → optional CORAL → UMAP). Once aligned,
we normalize the vectors, compute cosine similarities, and keep the top-K
song matches for every poem that pass a similarity (or distance) threshold.
The resulting matches are written to disk for downstream inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import normalize

from modality_alignment import (  # type: ignore
    DATA_PROCESSED,
    POEM_EMB_PATH,
    SONG_EMB_PATH,
    build_dataframe,
    coral_align,
    load_metadata,
    per_modality_zscore,
    remove_modality_direction_lr,
    run_umap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-NN retrieval from poems to songs in the aligned embedding space."
    )
    parser.add_argument("--k", type=int, default=10, help="Number of nearest songs per poem.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Keep matches with cosine similarity >= threshold (set <0 to disable).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=-1.0,
        help="Keep matches with cosine distance <= threshold (set <0 to disable).",
    )
    parser.add_argument(
        "--apply-coral",
        action="store_true",
        help="Apply CORAL alignment (should match modality_alignment settings).",
    )
    parser.add_argument(
        "--umap-dim",
        type=int,
        default=32,
        help="UMAP dimensionality for the shared space (32 matches modality_alignment).",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=80,
        help="UMAP neighbor count (should mirror modality_alignment).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.05,
        help="UMAP min_dist hyperparameter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PROCESSED / "poem_song_knn_matches.json",
        help="Where to write the filtered matches (JSON).",
    )
    return parser.parse_args()


def _load_aligned_embeddings(
    apply_coral: bool,
    umap_dim: int,
    umap_neighbors: int,
    umap_min_dist: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if not SONG_EMB_PATH.exists() or not POEM_EMB_PATH.exists():
        raise FileNotFoundError(
            "MPNet embedding files not found. Run generate_mpnet_embeddings.py first."
        )
    songs_emb = np.load(SONG_EMB_PATH).astype(np.float32)
    poems_emb = np.load(POEM_EMB_PATH).astype(np.float32)
    songs_emb, poems_emb = per_modality_zscore(songs_emb, poems_emb)
    songs_emb, poems_emb = remove_modality_direction_lr(songs_emb, poems_emb)
    if apply_coral:
        poems_emb = coral_align(songs_emb, poems_emb)
    combined = np.vstack([songs_emb, poems_emb])
    aligned = run_umap(
        combined,
        n_components=umap_dim,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
    )
    song_count = len(songs_emb)
    return aligned[:song_count], aligned[song_count:], combined


def cosine_knn(
    poems: NDArray[np.float32],
    songs: NDArray[np.float32],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    if k <= 0:
        raise ValueError("k must be positive")
    poem_norm = normalize(poems)
    song_norm = normalize(songs)
    scores = poem_norm @ song_norm.T
    k_eff = min(k, song_norm.shape[0])
    top_idx = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    order = np.argsort(-top_scores, axis=1)
    idx_sorted = np.take_along_axis(top_idx, order, axis=1)
    scores_sorted = np.take_along_axis(top_scores, order, axis=1)
    return idx_sorted.astype(np.int64), scores_sorted.astype(np.float32)


def filter_matches(
    poem_indices: Sequence[int],
    top_idx: NDArray[np.int64],
    top_scores: NDArray[np.float32],
    song_df: pd.DataFrame,
    poem_df: pd.DataFrame,
    sim_threshold: float,
    dist_threshold: float,
) -> List[dict]:
    records: List[dict] = []
    use_sim = sim_threshold >= 0.0
    use_dist = dist_threshold >= 0.0
    for p_row_idx, (indices, scores) in enumerate(zip(top_idx, top_scores)):
        poem_row = poem_df.iloc[p_row_idx]
        matches = []
        for idx, score in zip(indices, scores):
            cosine_dist = 1.0 - float(score)
            if use_sim and float(score) < sim_threshold:
                continue
            if use_dist and cosine_dist > dist_threshold:
                continue
            song_row = song_df.iloc[int(idx)]
            matches.append(
                {
                    "song_index": int(song_row["index"]),
                    "title": song_row.get("title"),
                    "creator": song_row.get("creator"),
                    "cosine_similarity": float(score),
                    "cosine_distance": cosine_dist,
                }
            )
        if matches:
            records.append(
                {
                    "poem_index": int(poem_row["index"]),
                    "title": poem_row.get("title"),
                    "creator": poem_row.get("creator"),
                    "matches": matches,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    song_emb_aligned, poem_emb_aligned, _ = _load_aligned_embeddings(
        apply_coral=args.apply_coral,
        umap_dim=args.umap_dim,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    top_idx, top_scores = cosine_knn(poem_emb_aligned, song_emb_aligned, args.k)

    songs_meta, poems_meta = load_metadata()
    df = build_dataframe(songs_meta, poems_meta)
    song_df = df[df["modality"] == "song"].reset_index(drop=True)
    poem_df = df[df["modality"] == "poem"].reset_index(drop=True)

    matches = filter_matches(
        poem_indices=poem_df.index,
        top_idx=top_idx,
        top_scores=top_scores,
        song_df=song_df,
        poem_df=poem_df,
        sim_threshold=args.similarity_threshold,
        dist_threshold=args.distance_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matches, indent=2))
    kept = sum(len(item["matches"]) for item in matches)
    print(
        f"Wrote {len(matches)} poem entries ({kept} poem→song matches) to {args.output}."
    )


if __name__ == "__main__":
    main()
