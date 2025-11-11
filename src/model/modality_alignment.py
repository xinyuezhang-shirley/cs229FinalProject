"""
Pipeline to align poem and song embeddings so they share clusters.

Steps
-----
1. Load mpnet embeddings for songs/poems plus their metadata.
2. Per-modality z-score normalization to remove scale differences.
3. Remove the primary "modality direction" so embeddings focus on content.
4. Reduce with UMAP (cosine metric) to 32 dimensions.
5. Cluster with HDBSCAN to get soft cluster memberships.
6. Provide cosine top-k matching with optional cross-encoder reranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import hdbscan  # type: ignore
import numpy as np
import pandas as pd
import umap  # type: ignore
from numpy.typing import NDArray
from scipy.linalg import fractional_matrix_power
from sentence_transformers import CrossEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

SONG_META_PATH = DATA_PROCESSED / "combined_songs_large_fixed.json"
POEM_META_PATH = DATA_RAW / "poetrydb_poems.json"
SONG_EMB_PATH = DATA_PROCESSED / "mpnet_embeddings_songs.npy"
POEM_EMB_PATH = DATA_PROCESSED / "mpnet_embeddings_poems.npy"


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_metadata() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    songs = _load_json(SONG_META_PATH)["items"]
    poems = _load_json(POEM_META_PATH)["items"]
    return songs, poems


def per_modality_zscore(
    songs: NDArray[np.float32],
    poems: NDArray[np.float32],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    def _z(x: NDArray[np.float32]) -> NDArray[np.float32]:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-6
        return (x - mean) / std

    return _z(songs), _z(poems)


def remove_modality_direction_lr(
    songs: NDArray[np.float32],
    poems: NDArray[np.float32],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Project away the direction learned by a linear modality classifier."""
    combined = np.vstack([songs, poems])
    labels = np.concatenate(
        [np.zeros(len(songs), dtype=np.int32), np.ones(len(poems), dtype=np.int32)]
    )
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(combined, labels)
    direction = clf.coef_[0]
    norm = np.linalg.norm(direction)
    if norm == 0:
        return songs, poems
    unit = direction / norm
    songs_adj = songs - (songs @ unit)[:, None] * unit
    poems_adj = poems - (poems @ unit)[:, None] * unit
    return songs_adj, poems_adj


def coral_align(
    reference: NDArray[np.float32],
    target: NDArray[np.float32],
    epsilon: float = 1e-3,
) -> NDArray[np.float32]:
    """Align target covariance to reference covariance using CORAL."""
    ref_cov = np.cov(reference, rowvar=False) + epsilon * np.eye(reference.shape[1])
    tgt_cov = np.cov(target, rowvar=False) + epsilon * np.eye(target.shape[1])
    ref_sqrt = fractional_matrix_power(ref_cov, 0.5).real
    tgt_inv_sqrt = fractional_matrix_power(tgt_cov, -0.5).real
    target_centered = target - target.mean(axis=0, keepdims=True)
    aligned = target_centered @ tgt_inv_sqrt @ ref_sqrt
    aligned += reference.mean(axis=0, keepdims=True)
    return aligned.astype(np.float32)


def run_umap(
    embeddings: NDArray[np.float32],
    n_components: int = 32,
    metric: str = "cosine",
    n_neighbors: int = 80,
    min_dist: float = 0.05,
) -> NDArray[np.float32]:
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(
    emb_32d: NDArray[np.float32],
    min_cluster_size: int = 5,
    min_samples: int = 5,
) -> hdbscan.HDBSCAN:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(emb_32d)
    return clusterer


def _resolve_kmeans_n_init() -> Union[int, str]:
    try:
        KMeans(n_clusters=2, n_init="auto")
        return "auto"
    except TypeError:  # pragma: no cover - older sklearn
        return 10


def run_kmeans_clustering(
    emb_32d: NDArray[np.float32],
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[NDArray[np.int32], NDArray[np.float32], KMeans]:
    if n_clusters < 2:
        raise ValueError("k-means requires at least 2 clusters")
    n_init = _resolve_kmeans_n_init()
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(emb_32d).astype(np.int32)
    distances = model.transform(emb_32d)
    min_dist = distances[np.arange(distances.shape[0]), labels]
    span = float(min_dist.max() - min_dist.min())
    if span <= 1e-8:
        probs = np.ones_like(min_dist, dtype=np.float32)
    else:
        probs = 1.0 - (min_dist - min_dist.min()) / span
    return labels, probs.astype(np.float32), model


def tune_kmeans_k(
    emb_32d: NDArray[np.float32],
    k_values: Sequence[int],
    metric: str = "euclidean",
    sample_size: Optional[int] = 2000,
    random_state: int = 42,
) -> Tuple[
    Optional[int],
    Optional[NDArray[np.int32]],
    Optional[NDArray[np.float32]],
    Optional[KMeans],
    pd.DataFrame,
]:
    rows: List[Dict[str, float]] = []
    best_score = -1.0
    best_k: Optional[int] = None
    best_labels: Optional[NDArray[np.int32]] = None
    best_probs: Optional[NDArray[np.float32]] = None
    best_model: Optional[KMeans] = None
    sample = None if (sample_size is None or sample_size <= 0) else sample_size
    for k in k_values:
        if k < 2 or k >= len(emb_32d):
            continue
        labels, probs, model = run_kmeans_clustering(emb_32d, k, random_state=random_state)
        score = silhouette_score(
            emb_32d,
            labels,
            metric=metric,
            sample_size=sample,
            random_state=random_state,
        )
        rows.append({"k": k, "silhouette": float(score)})
        if score > best_score:
            best_score = float(score)
            best_k = k
            best_labels = labels
            best_probs = probs
            best_model = model
    results = pd.DataFrame(rows)
    return best_k, best_labels, best_probs, best_model, results


def build_dataframe(
    songs: List[Dict[str, object]],
    poems: List[Dict[str, object]],
) -> pd.DataFrame:
    song_rows = [
        {
            "modality": "song",
            "index": idx,
            "title": item.get("title"),
            "creator": item.get("artist") or item.get("spotify_artist_name"),
            "text": item.get("lyrics", ""),
            "popularity": item.get("popularity"),
        }
        for idx, item in enumerate(songs)
    ]
    poem_rows = [
        {
            "modality": "poem",
            "index": idx,
            "title": item.get("title"),
            "creator": item.get("author"),
            "text": "\n".join(item.get("lines", [])),
            "linecount": item.get("linecount"),
        }
        for idx, item in enumerate(poems)
    ]
    return pd.DataFrame(song_rows + poem_rows)


def cosine_top_k(
    query_vectors: NDArray[np.float32],
    candidate_vectors: NDArray[np.float32],
    top_k: int = 5,
) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Return top-k candidate indices and scores sorted by cosine similarity."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_norm = normalize(query_vectors)
    candidate_norm = normalize(candidate_vectors)
    scores = query_norm @ candidate_norm.T
    k = min(top_k, candidate_vectors.shape[0])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-top_scores, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    scores_sorted = np.take_along_axis(top_scores, order, axis=1)
    return idx_sorted, scores_sorted


PUBLIC_CROSS_ENCODERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/stsb-roberta-large",
]


def _load_cross_encoder(model_name: str, max_length: int) -> CrossEncoder:
    errors = []
    candidates = PUBLIC_CROSS_ENCODERS if model_name == "auto" else [model_name]
    for name in candidates:
        try:
            print(f"Loading cross-encoder: {name}")
            return CrossEncoder(name, max_length=max_length)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append((name, exc))
            print(f"  ! Failed to load {name}: {exc}")
            continue
    raise RuntimeError(
        "Unable to load any cross-encoder:\n" + "\n".join(f"- {n}: {e}" for n, e in errors)
    )


def rerank_with_cross_encoder(
    model_name: str,
    queries: Sequence[str],
    candidates: Sequence[str],
    candidate_indices: NDArray[np.int64],
    max_length: int = 512,
) -> List[List[Tuple[int, float]]]:
    cross_encoder = _load_cross_encoder(model_name, max_length=max_length)
    reranked: List[List[Tuple[int, float]]] = []
    for q_text, indices in zip(queries, candidate_indices):
        pairs = [[q_text, candidates[idx]] for idx in indices]
        scores = cross_encoder.predict(pairs)
        ordered = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        reranked.append(ordered)
    return reranked


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["cluster", "modality"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"song": "song_count", "poem": "poem_count"})
    )
    summary["total"] = summary.sum(axis=1)
    summary["poem_share"] = summary["poem_count"] / summary["total"]
    return summary.sort_values("total", ascending=False)


def export_labels(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cluster labels to {path}")


def modality_leakage_report(
    features: NDArray[np.float32],
    modality_labels: NDArray[np.int32],
    cluster_labels: NDArray[np.int32],
) -> None:
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(features, modality_labels)
    probs = clf.predict_proba(features)[:, 1]
    preds = (probs >= 0.5).astype(np.int32)
    acc = accuracy_score(modality_labels, preds)
    auc = roc_auc_score(modality_labels, probs)
    mask = cluster_labels >= 0
    if mask.any():
        nmi = normalized_mutual_info_score(modality_labels[mask], cluster_labels[mask])
        ari = adjusted_rand_score(modality_labels[mask], cluster_labels[mask])
    else:
        nmi = float("nan")
        ari = float("nan")
    print(
        f"[Leakage] Logistic acc={acc:.3f}, AUC={auc:.3f}, "
        f"NMI={nmi:.3f}, ARI={ari:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align poem/song embeddings and run clustering.")
    parser.add_argument("--cluster-method", choices=["hdbscan", "kmeans"], default="hdbscan")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--kmeans-k", type=int, default=5, help="Number of clusters when using k-means.")
    parser.add_argument(
        "--kmeans-tune-range",
        default="",
        help="Comma-separated k values to evaluate via silhouette; overrides --kmeans-k when provided.",
    )
    parser.add_argument(
        "--kmeans-metric",
        default="euclidean",
        help="Distance metric for silhouette scoring during k tuning.",
    )
    parser.add_argument(
        "--kmeans-sample-size",
        type=int,
        default=2000,
        help="Sample size for silhouette scoring (<=0 means use every point).",
    )
    parser.add_argument(
        "--kmeans-random-state",
        type=int,
        default=42,
        help="Random seed for k-means clustering and tuning.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-K cosine matches before rerank.")
    parser.add_argument(
        "--cross-encoder",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder for reranking (use 'auto' to try multiple public models).",
    )
    parser.add_argument(
        "--cross-encoder-max-length",
        type=int,
        default=512,
        help="Max sequence length for the cross-encoder reranker.",
    )
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking.")
    parser.add_argument(
        "--apply-coral",
        action="store_true",
        help="Apply CORAL alignment after modality debiasing.",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=DATA_PROCESSED / "aligned_cluster_labels.csv",
        help="Where to write the merged dataframe with cluster labels for downstream notebooks.",
    )
    return parser.parse_args()


def _parse_k_values(spec: str) -> List[int]:
    if not spec:
        return []
    values: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"Invalid k value '{chunk}' in --kmeans-tune-range") from exc
    return values


def main() -> None:
    args = parse_args()

    if not SONG_EMB_PATH.exists() or not POEM_EMB_PATH.exists():
        raise FileNotFoundError(
            "mpnet embeddings not found. Run generate_mpnet_embeddings.py first."
        )

    songs_emb = np.load(SONG_EMB_PATH).astype(np.float32)
    poems_emb = np.load(POEM_EMB_PATH).astype(np.float32)

    songs_emb, poems_emb = per_modality_zscore(songs_emb, poems_emb)
    songs_emb, poems_emb = remove_modality_direction_lr(songs_emb, poems_emb)
    if args.apply_coral:
        poems_emb = coral_align(songs_emb, poems_emb)

    combined = np.vstack([songs_emb, poems_emb])
    modality_labels = np.concatenate(
        [np.zeros(len(songs_emb), dtype=np.int32), np.ones(len(poems_emb), dtype=np.int32)]
    )
    umap_emb = run_umap(combined)
    kmeans_tune_range = _parse_k_values(args.kmeans_tune_range)
    tuning_results: Optional[pd.DataFrame] = None
    if args.cluster_method == "hdbscan":
        clusterer = run_hdbscan(
            umap_emb,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        labels = clusterer.labels_
        probs = clusterer.probabilities_
    else:
        if kmeans_tune_range:
            best_k, best_labels, best_probs, _, tuning_results = tune_kmeans_k(
                umap_emb,
                kmeans_tune_range,
                metric=args.kmeans_metric,
                sample_size=args.kmeans_sample_size,
                random_state=args.kmeans_random_state,
            )
            if best_k is None or best_labels is None or best_probs is None:
                print("No valid k from tuning; falling back to --kmeans-k")
                labels, probs, _ = run_kmeans_clustering(
                    umap_emb,
                    args.kmeans_k,
                    random_state=args.kmeans_random_state,
                )
                chosen_k = args.kmeans_k
            else:
                labels = best_labels
                probs = best_probs
                chosen_k = best_k
        else:
            labels, probs, _ = run_kmeans_clustering(
                umap_emb,
                args.kmeans_k,
                random_state=args.kmeans_random_state,
            )
            chosen_k = args.kmeans_k
        print(f"k-means clustering complete (k={chosen_k}).")

    songs_meta, poems_meta = load_metadata()
    df = build_dataframe(songs_meta, poems_meta)
    df["cluster"] = labels
    df["cluster_prob"] = probs

    summary = summarize_clusters(df[df["cluster"] >= 0])
    print("Top clusters (mixed modalities expected):")
    print(summary.head(10))
    if tuning_results is not None and not tuning_results.empty:
        print("\nSilhouette sweep (top 10):")
        print(
            tuning_results.sort_values("silhouette", ascending=False)
            .head(10)
            .to_string(index=False)
        )
    modality_leakage_report(umap_emb, modality_labels, labels)
    export_labels(
        df[
            [
                "modality",
                "index",
                "title",
                "creator",
                "cluster",
                "cluster_prob",
            ]
        ],
        args.labels_output,
    )

    song_count = len(songs_emb)
    songs_32d = umap_emb[:song_count]
    poems_32d = umap_emb[song_count:]

    # Retrieval: poems to songs
    top_k_idx, top_k_scores = cosine_top_k(poems_32d, songs_32d, top_k=args.top_k)
    poem_rows = df[df["modality"] == "poem"].reset_index(drop=True)
    song_rows = df[df["modality"] == "song"].reset_index(drop=True)
    poem_texts = poem_rows["text"].tolist()
    song_texts = song_rows["text"].tolist()
    song_titles = song_rows["title"].tolist()
    song_creators = song_rows["creator"].tolist()

    if args.rerank:
        reranked = rerank_with_cross_encoder(
            args.cross_encoder,
            poem_texts[:5],
            song_texts,
            top_k_idx[:5],
            max_length=args.cross_encoder_max_length,
        )
        print("Sample reranked matches for first 5 poems:")
        for i, pairs in enumerate(reranked):
            matches = [
                {
                    "title": song_titles[idx],
                    "artist": song_creators[idx],
                    "score": float(score),
                }
                for idx, score in pairs[:5]
            ]
            print(f"Poem #{i}: {matches}")
    else:
        print("Sample cosine top-k matches (sorted):")
        for i in range(min(3, len(poem_rows))):
            matches = [
                {
                    "title": song_titles[idx],
                    "artist": song_creators[idx],
                    "cosine": float(score),
                }
                for idx, score in zip(top_k_idx[i][:5], top_k_scores[i][:5])
            ]
            print(f"Poem #{i}: {matches}")
        print("Enable --rerank to refine matches with a cross-encoder.")


if __name__ == "__main__":
    main()
