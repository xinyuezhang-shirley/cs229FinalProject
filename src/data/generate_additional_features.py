"""
Extract structural + semantic features for poems and songs.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Dict

import numpy as np
import pronouncing
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POEMS_PATH = PROJECT_ROOT / "data/raw/poetrydb_poems.json"
DEFAULT_SONGS_PATH = PROJECT_ROOT / "data/processed/combined_songs_large_fixed.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/semantic_features.npz"

PUNCT_RE = re.compile(r"[^A-Za-z0-9\s]")
SPACE_RE = re.compile(r"\s+")
SECTION_RE = re.compile(r"\[(Verse \d+|Verse|Pre-Chorus|Post-Chorus|Chorus|Intro|Bridge|Outro|Hook)[^\]]*\]", flags=re.IGNORECASE)

EMOTIONS = ["joy","sadness","anger","fear","love","nostalgia","longing","hope","excitement"]
THEMES = ["love/romance","heartbreak","nature","death","memory","religion","war/conflict","childhood","freedom","loneliness"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--poems-path", type=Path, default=DEFAULT_POEMS_PATH)
    p.add_argument("--songs-path", type=Path, default=DEFAULT_SONGS_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--lines-to-check", type=int, default=4)
    return p.parse_args()


def load_poems(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)["items"]


def load_songs(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)["items"]


def clean_line(t: str):
    t = PUNCT_RE.sub("", t)
    t = SPACE_RE.sub(" ", t)
    return t.strip().lower()


def clean_poem_lines(poems):
    out = []
    titles = []
    authors = []
    for poem in poems:
        lines = []
        for x in poem.get("lines", []):
            y = clean_line(x)
            if y:
                lines.append(y)
        out.append(lines)
        titles.append(poem.get("title", ""))
        authors.append(poem.get("author", ""))
    return out, titles, authors


def clean_song_lyrics(songs):
    out = []
    durations = []
    titles = []
    artists = []
    idxs = []

    for i, song in enumerate(songs):
        lyr = (song.get("lyrics") or "").replace("Contributors\nTranslations", "")
        lyr = SECTION_RE.sub(" ", lyr)
        raw = lyr.split("\n")

        lines = []
        for x in raw:
            y = clean_line(x)
            if y:
                lines.append(y)

        if not lines:
            continue

        out.append(lines)
        durations.append((song.get("duration_ms") or 0) / 60000)
        titles.append(song.get("title", ""))
        artists.append(song.get("artist") or song.get("spotify_artist_name", ""))
        idxs.append(i)

    return out, durations, titles, artists, idxs


@lru_cache(maxsize=4096)
def _phones(w: str):
    return tuple(pronouncing.phones_for_word(w))


@lru_cache(maxsize=4096)
def _rhymes(w: str):
    return frozenset(pronouncing.rhymes(w))


def words_rhyme(a: str, b: str):
    if _phones(a) and _phones(b):
        return b in _rhymes(a)
    if len(a) > 1 and len(b) > 1:
        return a[-2:] == b[-2:]
    return a[-1:] == b[-1:]


def rhyme_density(lines, L: int):
    ends = [ln.split()[-1] for ln in lines if ln.split()]
    if len(ends) < 2:
        return 0.0

    hits = 0
    total = 0
    for i, w in enumerate(ends):
        for off in range(1, L + 1):
            j = i + off
            if j >= len(ends):
                break
            total += 1
            if words_rhyme(w, ends[j]):
                hits += 1

    if total == 0:
        return 0.0
    return hits / total


def word_stats(lines):
    n = len(lines)
    if n == 0:
        return 0, 0.0, 0
    counts = [len(ln.split()) for ln in lines]
    total = sum(counts)
    return n, total / n, total


def make_poem_struct(lines, L):
    arr = []
    for Ls in lines:
        n, avg, _ = word_stats(Ls)
        arr.append([avg, n, rhyme_density(Ls, L)])
    return np.array(arr)


def make_song_struct(lines, durations, L):
    arr = []
    for Ls, d in zip(lines, durations):
        n, avg, tot = word_stats(Ls)
        wpm = tot / d if d > 0 else 0.0
        arr.append([avg, n, wpm, rhyme_density(Ls, L)])
    return np.array(arr)


def get_semantic(text: str, clf):
    emo = clf(text, EMOTIONS, multi_label=True)["scores"]
    thm = clf(text, THEMES, multi_label=True)["scores"]
    return np.array(emo + thm, dtype=np.float32)


def batch_semantic(texts, clf):
    return np.vstack([get_semantic(t, clf) for t in texts])


def main():
    args = parse_args()

    clf = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

    poems = load_poems(args.poems_path)
    poem_lines, poem_titles, poem_authors = clean_poem_lines(poems)
    poem_texts = [" ".join(x) for x in poem_lines]

    songs = load_songs(args.songs_path)
    song_lines, song_durs, song_titles, song_artists, song_idxs = clean_song_lyrics(songs)
    song_texts = [" ".join(x) for x in song_lines]

    poem_struct_raw = make_poem_struct(poem_lines, args.lines_to_check)
    song_struct_raw = make_song_struct(song_lines, song_durs, args.lines_to_check)

    poem_struct = (poem_struct_raw - poem_struct_raw.mean(0)) / poem_struct_raw.std(0)
    song_struct = (song_struct_raw - song_struct_raw.mean(0)) / song_struct_raw.std(0)

    poem_sem = batch_semantic(poem_texts, clf)
    song_sem = batch_semantic(song_texts, clf)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        poem_struct_raw=poem_struct_raw,
        poem_struct=poem_struct,
        poem_semantic=poem_sem,
        poem_titles=np.array(poem_titles),
        poem_authors=np.array(poem_authors),
        song_struct_raw=song_struct_raw,
        song_struct=song_struct,
        song_semantic=song_sem,
        song_titles=np.array(song_titles),
        song_artists=np.array(song_artists),
        song_source_indexes=np.array(song_idxs),
        emotion_labels=np.array(EMOTIONS),
        theme_labels=np.array(THEMES),
    )


if __name__ == "__main__":
    main()
