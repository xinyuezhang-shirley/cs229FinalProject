"""Utilities to download poems from PoetryDB and cache them locally."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import requests

MAX_RETRIES = 6
BACKOFF_SECONDS = 2.0
REQUEST_DELAY = 1.0

_SESSION = requests.Session()


def _get_with_retry(url: str) -> requests.Response:
    """Fetch a URL with simple exponential backoff retry logic."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _SESSION.get(url, timeout=30)
            if response.status_code >= 500:
                response.raise_for_status()
            return response
        except Exception as exc:  # broad to retry on network hiccups
            last_error = exc
            sleep_for = BACKOFF_SECONDS * (2 ** attempt)
            print(f"Request failed ({exc}); retrying in {sleep_for:.1f}s...")
            time.sleep(sleep_for)
    assert last_error is not None
    raise last_error


BASE_URL = "https://poetrydb.org"
RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "poetrydb_poems.json"

# Authors to skip due to API issues (too many poems, API times out)
SKIP_AUTHORS = set()  # type: set[str]
# Authors that need special title-by-title fetching
PROLIFIC_AUTHORS = {"George Gordon, Lord Byron", "Percy Bysshe Shelley"}


def fetch_authors() -> List[str]:
    """Return the list of authors available in PoetryDB."""
    response = _get_with_retry(f"{BASE_URL}/author")
    response.raise_for_status()
    payload = response.json()
    authors = payload.get("authors")
    if not authors:
        raise RuntimeError("PoetryDB response missing 'authors' key")
    return authors


def fetch_poems_by_author(author: str) -> List[Dict[str, object]]:
    """Return all poems for a given author."""
    encoded_author = quote(author)
    response = _get_with_retry(f"{BASE_URL}/author/{encoded_author}")
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("title") == "Not Found":
        return []
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected payload for author {author!r}: {payload}")
    return payload


def fetch_poems_by_author_via_titles(author: str) -> List[Dict[str, object]]:
    """Fetch poems for prolific authors by getting titles first, then individual poems."""
    encoded_author = quote(author)
    print(f"    Using title-by-title method for {author}...")
    
    # Step 1: Get just the titles (lightweight)
    response = _get_with_retry(f"{BASE_URL}/author/{encoded_author}/title")
    response.raise_for_status()
    payload = response.json()
    
    if isinstance(payload, dict) and payload.get("title") == "Not Found":
        return []
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected payload for author {author!r}: {payload}")
    
    # Extract unique titles
    titles = {entry.get("title") for entry in payload if isinstance(entry, dict) and entry.get("title")}
    print(f"    Found {len(titles)} titles for {author}")
    
    poems: List[Dict[str, object]] = []
    
    # Step 2: Fetch each poem by author+title combo
    for idx, title in enumerate(sorted(filter(None, titles)), start=1):
        try:
            encoded_title = quote(title)
            # Use author,title search to get specific poem
            url = f"{BASE_URL}/author,title/{encoded_author};{encoded_title}"
            response = _get_with_retry(url)
            response.raise_for_status()
            poem_payload = response.json()
            
            if isinstance(poem_payload, list):
                # Filter to only poems by this author (in case of title collisions)
                for poem in poem_payload:
                    if isinstance(poem, dict) and poem.get("author") == author:
                        poems.append(poem)
            
            # Small delay to be polite to API
            if idx % 10 == 0:
                print(f"      Progress: {idx}/{len(titles)} titles fetched")
                time.sleep(REQUEST_DELAY)
            else:
                time.sleep(REQUEST_DELAY * 0.5)
                
        except Exception as exc:
            print(f"      Failed to fetch '{title}': {exc}")
            continue
    
    return poems


def download_all_poems(output_path: Path = RAW_DATA_PATH) -> Path:
    """Download the complete PoetryDB corpus and save it as JSON."""
    # Load existing progress if available
    poems: List[Dict[str, object]] = []
    failed_authors: dict[str, str] = {}
    completed_authors: set[str] = set()
    
    if output_path.exists():
        print(f"Loading existing data from {output_path}...")
        with output_path.open("r", encoding="utf-8") as fp:
            existing = json.load(fp)
            poems = existing.get("items", [])
            failed_authors = existing.get("failed_authors", {})
            # Track which authors we already have
            for poem in poems:
                if "author" in poem:
                    completed_authors.add(poem["author"])
        print(f"Resuming: {len(poems)} poems from {len(completed_authors)} authors already collected")
    
    authors = fetch_authors()
    print(f"Found {len(authors)} authors; fetching poems...")

    for idx, author in enumerate(authors, start=1):
        if author in failed_authors:
            print(f"[{idx}/{len(authors)}] {author}: skipping (previously failed)")
            continue
        
        if author in completed_authors:
            print(f"[{idx}/{len(authors)}] {author}: skipping (already processed)")
            continue
        
        if author in SKIP_AUTHORS:
            print(f"[{idx}/{len(authors)}] {author}: skipping (known API issue)")
            failed_authors[author] = "Skipped due to API timeout/rate limit issues"
            continue
            
        try:
            # Use special method for prolific authors
            if author in PROLIFIC_AUTHORS:
                author_poems = fetch_poems_by_author_via_titles(author)
            else:
                author_poems = fetch_poems_by_author(author)
            poems.extend(author_poems)
            completed_authors.add(author)
            count = len(author_poems)
            print(f"[{idx}/{len(authors)}] {author}: {count} poems")
            
            # Save progress every 5 authors
            if idx % 5 == 0:
                _save_dataset(output_path, poems, authors, failed_authors)
                
        except Exception as exc:  # keep moving if an author fetch fails
            failed_authors[author] = str(exc)
            print(f"[{idx}/{len(authors)}] {author}: failed ({exc})")
        
        time.sleep(REQUEST_DELAY)

    # Final save
    _save_dataset(output_path, poems, authors, failed_authors)
    print(f"✓ Saved {len(poems)} poems to {output_path}")
    return output_path


def _save_dataset(
    output_path: Path, 
    poems: List[Dict[str, object]], 
    authors: List[str], 
    failed_authors: dict[str, str]
) -> None:
    """Save current progress to disk."""
    dataset = {
        "source": "PoetryDB",
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "author_count": len(authors),
        "poem_count": len(poems),
        "failed_authors": failed_authors,
        "items": poems,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(dataset, fp, indent=2)
    print(f"  → Progress saved: {len(poems)} poems")


if __name__ == "__main__":
    download_all_poems()
