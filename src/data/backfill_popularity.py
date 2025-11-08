"""Backfill missing popularity values in existing song data.

This script loads a songs JSON file and fetches missing popularity scores
from Spotify by searching for each song.
"""
from __future__ import annotations
import os
import json
import time
from pathlib import Path
from difflib import SequenceMatcher

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise ValueError("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file")


def normalize_title(title: str) -> str:
    t = (title or "").lower().strip()
    t = t.replace("'", "")
    for suf in [" - remaster", " - remastered", " (remaster)", " (remastered)",
                " - radio edit", " (radio edit)", " - single version", " - live", " (live)"]:
        t = t.replace(suf, "")
    t = " ".join(t.split())
    return t


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def main():
    input_file = project_root / 'data' / 'processed' / 'combined_songs_large.json'
    output_file = project_root / 'data' / 'processed' / 'combined_songs_large_fixed.json'
    
    print(f"Loading {input_file}...")
    with input_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    missing_count = sum(1 for s in items if s.get('popularity') is None)
    print(f"Total songs: {len(items)}")
    print(f"Songs missing popularity: {missing_count}")
    
    if missing_count == 0:
        print("All songs already have popularity!")
        return
    
    creds = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=creds)
    
    fixed = 0
    failed = 0
    
    for i, song in enumerate(items, 1):
        if song.get('popularity') is not None:
            continue
        
        # Try to find the track on Spotify
        query = f"track:{song.get('title', '')} artist:{song.get('artist', '')}"
        try:
            results = sp.search(q=query, type='track', limit=5)
            tracks = results.get('tracks', {}).get('items', [])
            
            # Find best match
            best_match = None
            best_score = 0.0
            for t in tracks:
                title_sim = similarity(song.get('title', ''), t.get('name', ''))
                artist_sim = similarity(song.get('artist', ''), t.get('artists', [{}])[0].get('name', ''))
                combined_score = (title_sim + artist_sim) / 2
                if combined_score > best_score and combined_score >= 0.7:
                    best_match = t
                    best_score = combined_score
            
            if best_match and best_match.get('popularity') is not None:
                song['popularity'] = best_match['popularity']
                fixed += 1
                if fixed % 10 == 0:
                    print(f"  Fixed {fixed}/{missing_count} songs...")
            else:
                failed += 1
            
            time.sleep(0.1)  # rate limiting
            
        except Exception as e:
            failed += 1
            if failed % 50 == 0:
                print(f"  [{i}/{len(items)}] Failed to fetch: {failed} songs")
    
    print(f"\nBackfill complete!")
    print(f"  Fixed: {fixed}")
    print(f"  Failed: {failed}")
    print(f"  Remaining missing: {sum(1 for s in items if s.get('popularity') is None)}")
    
    # Save updated data
    with output_file.open('w', encoding='utf-8') as f:
        json.dump({'source': data.get('source', 'Spotify + Genius'), 
                   'total_songs': len(items), 
                   'items': items}, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    main()
