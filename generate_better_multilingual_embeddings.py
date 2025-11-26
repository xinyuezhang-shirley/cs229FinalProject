"""
Multilingual embeddings using paraphrase-multilingual-mpnet-base-v2
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import re
from functools import reduce

print("Loading data...")

# Load poems
with open("data/raw/poetrydb_poems.json", "r") as f:
    poem_data = json.load(f)

# Load songs
with open("data/processed/combined_songs_large_fixed.json", "r") as f:
    song_data = json.load(f)

print("Cleaning poem texts...")

# Clean poems (same as test_vectorization.ipynb)
poems_and_info = poem_data['items']
raw_poem_texts = [" ".join(poem["lines"]) for poem in poems_and_info]

poem_texts = [
    re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', poem.lower())).strip()
    for poem in raw_poem_texts
]

print(f"Loaded {len(poem_texts)} poems")

print("Cleaning song texts...")

# Clean songs (same as test_vectorization.ipynb)
songs_and_info = song_data['items']
raw_song_texts = [song["lyrics"] for song in songs_and_info]

replacements_space = [
    r"\[Verse \d+\]",
    r"\[Chorus\]",
    r"\r?\n",
]

punctuation_pattern = r"[^A-Za-z0-9\s]"

song_texts = [
    re.sub(
        punctuation_pattern, "",
        re.sub(
            r"\s+", " ",
            reduce(lambda text, pat: re.sub(pat, " ", text), replacements_space, song)
        )
    ).strip().lower()
    for song in raw_song_texts
]

print(f"Loaded {len(song_texts)} songs")

print("\nLoading paraphrase-multilingual-mpnet-base-v2 model...")
print("(This is a larger, better quality multilingual model - 768 dimensions)")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("\nEncoding poems... (this may take a few minutes)")
mpnet_embeddings_poems = model.encode(poem_texts, normalize_embeddings=True, show_progress_bar=True)

print("\nEncoding songs... (this may take a few minutes)")
mpnet_embeddings_songs = model.encode(song_texts, normalize_embeddings=True, show_progress_bar=True)

print(f"\nPoem embeddings shape: {mpnet_embeddings_poems.shape}")
print(f"Song embeddings shape: {mpnet_embeddings_songs.shape}")

# Save to data/processed
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

print("\nSaving embeddings...")
np.save(processed_dir / "mpnet_embeddings_poems.npy", mpnet_embeddings_poems)
np.save(processed_dir / "mpnet_embeddings_songs.npy", mpnet_embeddings_songs)

print(f"\n✓ Saved to:")
print(f"  - {processed_dir / 'mpnet_embeddings_poems.npy'}")
print(f"  - {processed_dir / 'mpnet_embeddings_songs.npy'}")

print("\nTo load these embeddings later:")
print("  import numpy as np")
print("  poems = np.load('data/processed/mpnet_embeddings_poems.npy')")
print("  songs = np.load('data/processed/mpnet_embeddings_songs.npy')")
