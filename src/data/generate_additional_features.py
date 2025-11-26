"""Generate auxiliary poem and song features used for downstream modeling."""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import pronouncing
except ImportError as exc:  # pragma: no cover - clearer error message
    raise ImportError(
        "Missing dependency 'pronouncing'. Install it with `pip install pronouncing`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POEMS_PATH = PROJECT_ROOT / "data" / "raw" / "poetrydb_poems.json"
DEFAULT_SONGS_PATH = PROJECT_ROOT / "data" / "processed" / "combined_songs_large_fixed.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "additional_features.npz"

PUNCTUATION_RE = re.compile(r"[^A-Za-z0-9\s]")
SPACE_RE = re.compile(r"\s+")
SECTION_LABEL_RE = re.compile(
    r"\[(Verse \d+|Verse|Pre-Chorus|Post-Chorus|Chorus|Intro|Bridge|Outro|Hook)[^\]]*\]",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute additional features (structure + rhyme) for poems and songs."
    )
    parser.add_argument(
        "--poems-path",
        type=Path,
        default=DEFAULT_POEMS_PATH,
        help=f"Path to poetry JSON (default: {DEFAULT_POEMS_PATH})",
    )
    parser.add_argument(
        "--songs-path",
        type=Path,
        default=DEFAULT_SONGS_PATH,
        help=f"Path to song lyrics JSON (default: {DEFAULT_SONGS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination .npz file for serialized features (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--lines-to-check",
        type=int,
        default=4,
        help="Number of forward lines to use when computing rhyme density.",
    )
    return parser.parse_args()


def load_poems(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["items"]


def load_songs(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["items"]


def clean_text_line(text: str) -> str:
    text = PUNCTUATION_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip().lower()


def clean_poem_lines(poems: Sequence[Dict]) -> Tuple[List[List[str]], List[str], List[str]]:
    cleaned_lines: List[List[str]] = []
    titles: List[str] = []
    authors: List[str] = []

    for poem in poems:
        lines = []
        for raw_line in poem.get("lines", []):
            cleaned = clean_text_line(raw_line)
            if cleaned:
                lines.append(cleaned)

        cleaned_lines.append(lines)
        titles.append(poem.get("title", ""))
        authors.append(poem.get("author", ""))

    return cleaned_lines, titles, authors


def clean_song_lyrics(songs: Sequence[Dict]) -> Tuple[List[List[str]], List[float], List[str], List[str], List[int]]:
    cleaned_lines: List[List[str]] = []
    durations_min: List[float] = []
    titles: List[str] = []
    artists: List[str] = []
    original_indexes: List[int] = []

    for idx, song in enumerate(songs):
        lyrics = song.get("lyrics", "") or ""

        if "Contributors\nTranslations" in lyrics:
            continue

        lyrics = SECTION_LABEL_RE.sub(" ", lyrics)
        raw_lines = lyrics.split("\n")

        song_lines = []
        for raw_line in raw_lines:
            cleaned = clean_text_line(raw_line)
            if cleaned:
                song_lines.append(cleaned)

        if not song_lines:
            continue

        cleaned_lines.append(song_lines)
        durations_min.append(max((song.get("duration_ms") or 0) / 60000, 0))
        titles.append(song.get("title", ""))
        artists.append(song.get("artist") or song.get("spotify_artist_name", ""))
        original_indexes.append(idx)

    return cleaned_lines, durations_min, titles, artists, original_indexes


@lru_cache(maxsize=4096)
def _phones(word: str) -> Tuple[str, ...]:
    return tuple(pronouncing.phones_for_word(word))


@lru_cache(maxsize=4096)
def _rhyme_set(word: str) -> frozenset:
    return frozenset(pronouncing.rhymes(word))


def words_rhyme(word_a: str, word_b: str) -> bool:
    if _phones(word_a) and _phones(word_b):
        return word_b in _rhyme_set(word_a)

    if len(word_a) >= 2 and len(word_b) >= 2:
        return word_a[-2:] == word_b[-2:]

    return word_a[-1:] == word_b[-1:]


def compute_rhyme_density(lines: Sequence[str], lines_to_check: int) -> float:
    last_words = []
    for line in lines:
        tokens = line.split()
        if tokens:
            last_words.append(tokens[-1])

    if len(last_words) < 2:
        return 0.0

    rhyme_counts = 0
    comparisons = 0

    for i, word in enumerate(last_words):
        for offset in range(1, lines_to_check + 1):
            j = i + offset
            if j >= len(last_words):
                break

            comparisons += 1
            if words_rhyme(word, last_words[j]):
                rhyme_counts += 1

    if comparisons == 0:
        return 0.0

    return rhyme_counts / comparisons


def compute_word_stats(lines: Sequence[str]) -> Tuple[int, float, int]:
    num_lines = len(lines)
    if num_lines == 0:
        return 0, 0.0, 0

    word_counts = [len(line.split()) for line in lines]
    total_words = sum(word_counts)
    mean_words = total_words / num_lines if num_lines else 0.0
    return num_lines, mean_words, total_words


def make_poem_feature_matrix(lines: Sequence[Sequence[str]], lines_to_check: int) -> np.ndarray:
    mean_words: List[float] = []
    num_lines: List[int] = []
    rhyme_density: List[float] = []

    for poem_lines in lines:
        num, mean, _ = compute_word_stats(poem_lines)
        mean_words.append(mean)
        num_lines.append(num)
        rhyme_density.append(compute_rhyme_density(poem_lines, lines_to_check))

    return np.column_stack([mean_words, num_lines, rhyme_density])


def make_song_feature_matrix(
    lines: Sequence[Sequence[str]],
    durations_min: Sequence[float],
    lines_to_check: int,
) -> np.ndarray:
    mean_words: List[float] = []
    num_lines: List[int] = []
    wpm: List[float] = []
    rhyme_density: List[float] = []

    for song_lines, duration in zip(lines, durations_min):
        num, mean, total = compute_word_stats(song_lines)
        mean_words.append(mean)
        num_lines.append(num)
        if duration > 0:
            wpm.append(total / duration)
        else:
            wpm.append(0.0)
        rhyme_density.append(compute_rhyme_density(song_lines, lines_to_check))

    return np.column_stack([mean_words, num_lines, wpm, rhyme_density])


def standardize(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    standardized = (matrix - means) / stds
    return standardized, means, stds


def main() -> None:
    args = parse_args()

    if not args.poems_path.exists():
        raise FileNotFoundError(f"Cannot find poems JSON at {args.poems_path}")
    if not args.songs_path.exists():
        raise FileNotFoundError(f"Cannot find songs JSON at {args.songs_path}")

    poems = load_poems(args.poems_path)
    poem_line_sets, poem_titles, poem_authors = clean_poem_lines(poems)
    poem_features_raw = make_poem_feature_matrix(poem_line_sets, args.lines_to_check)
    poem_features, poem_means, poem_stds = standardize(poem_features_raw)

    songs = load_songs(args.songs_path)
    (
        song_line_sets,
        song_durations_min,
        song_titles,
        song_artists,
        song_original_indexes,
    ) = clean_song_lyrics(songs)
    song_features_raw = make_song_feature_matrix(
        song_line_sets, song_durations_min, args.lines_to_check
    )
    song_features, song_means, song_stds = standardize(song_features_raw)

    feature_names_poems = np.array(
        ["mean_words_per_line", "num_lines", "rhyme_density"]
    )
    feature_names_songs = np.array(
        ["mean_words_per_line", "num_lines", "words_per_minute", "rhyme_density"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        poems_raw=poem_features_raw,
        poems_standardized=poem_features,
        poems_feature_names=feature_names_poems,
        poems_means=poem_means,
        poems_stds=poem_stds,
        poems_titles=np.array(poem_titles),
        poems_authors=np.array(poem_authors),
        songs_raw=song_features_raw,
        songs_standardized=song_features,
        songs_feature_names=feature_names_songs,
        songs_means=song_means,
        songs_stds=song_stds,
        songs_titles=np.array(song_titles),
        songs_artists=np.array(song_artists),
        songs_source_indexes=np.array(song_original_indexes),
        lines_to_check=np.array(args.lines_to_check),
    )

    print(
        f"Saved poem ({poem_features.shape[0]}) and song ({song_features.shape[0]}) "
        f"feature matrices to {args.output}"
    )


if __name__ == "__main__":
    main()
