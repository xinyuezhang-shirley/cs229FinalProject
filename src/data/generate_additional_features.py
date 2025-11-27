from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Sequence, Dict
from functools import lru_cache

import numpy as np
import pronouncing
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POEMS_PATH = PROJECT_ROOT / "data/raw/poetrydb_poems.json"
DEFAULT_SONGS_PATH = PROJECT_ROOT / "data/processed/combined_songs_large_fixed.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/full_features.npz"

PUNCT_RE = re.compile(r"[^A-Za-z0-9\s]")
SPACE_RE = re.compile(r"\s+")
SECTION_RE = re.compile(
    r"\[(Verse \d+|Verse|Pre-Chorus|Post-Chorus|Chorus|Intro|Bridge|Outro|Hook)[^\]]*\]",
    flags=re.IGNORECASE
)

# original lists
EMOTIONS = ["joy","sadness","anger","fear","love","nostalgia","longing","hope","excitement"]
THEMES = ["love/romance","heartbreak","nature","death","memory","religion","war/conflict","childhood","freedom","loneliness"]

# added groups
SENTIMENT = ["positive","negative"]
SUBJECTIVITY = ["subjective","objective"]
CONCRETE = ["concrete imagery","abstract/figurative"]
ENERGY = ["calm","energetic","intense","slow"]
NARRATIVE = ["narrative/storytelling","reflective","dialogue-like"]
IMAGERY = ["visual imagery","auditory imagery","tactile imagery","emotional imagery"]

ALL_GROUPS = [
    EMOTIONS, THEMES, SENTIMENT, SUBJECTIVITY,
    CONCRETE, ENERGY, NARRATIVE, IMAGERY
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--poems-path", type=Path, default=DEFAULT_POEMS_PATH)
    p.add_argument("--songs-path", type=Path, default=DEFAULT_SONGS_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--lines-to-check", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def load_poems(p):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)["items"]


def load_songs(p):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)["items"]


def clean_line(t: str):
    t = PUNCT_RE.sub("", t)
    t = SPACE_RE.sub(" ", t)
    return t.strip().lower()


def clean_poem_lines(poems):
    lines = []
    titles = []
    authors = []
    for poem in poems:
        L = []
        for raw in poem.get("lines", []):
            s = clean_line(raw)
            if s:
                L.append(s)
        lines.append(L)
        titles.append(poem.get("title", ""))
        authors.append(poem.get("author", ""))
    return lines, titles, authors


def clean_song_lyrics(songs):
    lines = []
    durs = []
    titles = []
    artists = []
    idxs = []

    for i, song in enumerate(songs):
        lyr = (song.get("lyrics") or "")
        lyr = lyr.replace("Contributors\nTranslations", "")
        lyr = SECTION_RE.sub(" ", lyr)
        raw = lyr.split("\n")

        L = []
        for r in raw:
            s = clean_line(r)
            if s:
                L.append(s)

        if not L:
            continue

        lines.append(L)
        durs.append((song.get("duration_ms") or 0) / 60000)
        titles.append(song.get("title", ""))
        artists.append(song.get("artist") or song.get("spotify_artist_name", ""))
        idxs.append(i)

    return lines, durs, titles, artists, idxs


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
    tot = 0
    for i, w in enumerate(ends):
        for off in range(1, L + 1):
            j = i + off
            if j >= len(ends):
                break
            tot += 1
            if words_rhyme(w, ends[j]):
                hits += 1
    if tot == 0:
        return 0.0
    return hits / tot


def word_stats(lines):
    n = len(lines)
    if n == 0:
        return 0, 0.0, 0
    counts = [len(ln.split()) for ln in lines]
    tot = sum(counts)
    return n, tot / n, tot


def lexical_features(lines: List[str]):
    words = []
    for ln in lines:
        words.extend(ln.split())
    if not words:
        return np.array([0, 0, 0], dtype=float)

    vocab = set(words)
    vocab_size = len(vocab)
    ttr = vocab_size / len(words)
    avg_len = sum(len(w) for w in words) / len(words)
    return np.array([vocab_size, ttr, avg_len], dtype=float)


def make_poem_struct(lines, L):
    out = []
    for Ls in lines:
        n, avg, _ = word_stats(Ls)
        out.append([avg, n, rhyme_density(Ls, L)])
    return np.array(out)


def make_song_struct(lines, durs, L):
    out = []
    for Ls, d in zip(lines, durs):
        n, avg, tot = word_stats(Ls)
        wpm = tot / d if d > 0 else 0.0
        out.append([avg, n, wpm, rhyme_density(Ls, L)])
    return np.array(out)


def batch_semantic(texts, clf, groups, bs: int):
    feats = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        for t in chunk:
            vec = []
            for g in groups:
                out = clf(t, g, multi_label=True)["scores"]
                vec.extend(out)
            feats.append(vec)
    return np.array(feats, dtype=float)


def batch_lexical(lines):
    return np.vstack([lexical_features(x) for x in lines])


def main():
    args = parse_args()

    clf = pipeline("zero-shot-classification",
                   model="joeddav/xlm-roberta-large-xnli")

    poems = load_poems(args.poems_path)
    poem_lines, poem_titles, poem_authors = clean_poem_lines(poems)
    poem_texts = [" ".join(x) for x in poem_lines]

    songs = load_songs(args.songs_path)
    song_lines, song_durs, song_titles, song_artists, song_idxs =
        clean_song_lyrics(songs)
    song_texts = [" ".join(x) for x in song_lines]

    poem_struct_raw = make_poem_struct(poem_lines, args.lines_to_check)
    song_struct_raw = make_song_struct(song_lines, song_durs, args.lines_to_check)

    poem_struct = (poem_struct_raw - poem_struct_raw.mean(0)) / poem_struct_raw.std(0)
    song_struct = (song_struct_raw - song_struct_raw.mean(0)) / song_struct_raw.std(0)

    poem_sem = batch_semantic(poem_texts, clf, ALL_GROUPS, args.batch_size)
    song_sem = batch_semantic(song_texts, clf, ALL_GROUPS, args.batch_size)

    poem_lex = batch_lexical(poem_lines)
    song_lex = batch_lexical(song_lines)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        poem_struct_raw=poem_struct_raw,
        poem_struct=poem_struct,
        poem_semantic=poem_sem,
        poem_lexical=poem_lex,
        poem_titles=np.array(poem_titles),
        poem_authors=np.array(poem_authors),

        song_struct_raw=song_struct_raw,
        song_struct=song_struct,
        song_semantic=song_sem,
        song_lexical=song_lex,
        song_titles=np.array(song_titles),
        song_artists=np.array(song_artists),
        song_source_indexes=np.array(song_idxs),

        emotion_labels=np.array(EMOTIONS),
        theme_labels=np.array(THEMES),
    )


if __name__ == "__main__":
    main()
