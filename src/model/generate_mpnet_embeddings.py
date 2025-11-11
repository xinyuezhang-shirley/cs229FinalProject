from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATA_RAW = PROJECT_ROOT / "data" / "raw"

DEFAULT_SONGS_JSON = DEFAULT_DATA_PROCESSED / "combined_songs_large_fixed.json"
DEFAULT_POEMS_JSON = DEFAULT_DATA_RAW / "poetrydb_poems.json"
DEFAULT_SONGS_EMB = DEFAULT_DATA_PROCESSED / "mpnet_embeddings_songs.npy"
DEFAULT_POEMS_EMB = DEFAULT_DATA_PROCESSED / "mpnet_embeddings_poems.npy"


def load_song_lyrics(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    texts = []
    missing = 0
    for item in payload.get("items", []):
        lyrics = item.get("lyrics")
        if not lyrics:
            missing += 1
            texts.append("")
        else:
            texts.append(lyrics)
    if missing:
        print(f"[songs] Warning: {missing} items missing lyrics; encoded empty strings for them")
    return texts


def load_poem_texts(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    texts = []
    for poem in payload.get("items", []):
        lines = poem.get("lines") or []
        text = "\n".join(lines) if isinstance(lines, Sequence) else str(lines)
        texts.append(text)
    return texts


def encode_texts(model: SentenceTransformer, texts: Sequence[str], batch_size: int, normalize: bool) -> np.ndarray:
    if not texts:
        raise ValueError("No texts provided for encoding")
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )


def save_embeddings(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    print(f"Saved embeddings to {path} ({array.shape[0]} items, dim={array.shape[1]})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SentenceTransformer embeddings (all-mpnet-base-v2) for poems and songs.",
    )
    parser.add_argument("--songs-json", type=Path, default=DEFAULT_SONGS_JSON)
    parser.add_argument("--poems-json", type=Path, default=DEFAULT_POEMS_JSON)
    parser.add_argument("--songs-output", type=Path, default=DEFAULT_SONGS_EMB)
    parser.add_argument("--poems-output", type=Path, default=DEFAULT_POEMS_EMB)
    parser.add_argument("--model-name", default="all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization on embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize = not args.no_normalize

    print("Loading data...")
    song_texts = load_song_lyrics(args.songs_json)
    poem_texts = load_poem_texts(args.poems_json)
    print(f"Songs: {len(song_texts)} texts | Poems: {len(poem_texts)} texts")

    print(f"Loading SentenceTransformer model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("Encoding songs...")
    song_embeddings = encode_texts(model, song_texts, args.batch_size, normalize)
    print("Encoding poems...")
    poem_embeddings = encode_texts(model, poem_texts, args.batch_size, normalize)

    save_embeddings(args.songs_output, song_embeddings)
    save_embeddings(args.poems_output, poem_embeddings)


if __name__ == "__main__":
    main()
