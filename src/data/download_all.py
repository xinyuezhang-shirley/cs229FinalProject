"""Download and merge song data from both Spotify and Genius APIs."""
import json
import os
from pathlib import Path
import time
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Always load .env from project root
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

# Get API credentials
GENIUS_TOKEN = os.getenv("GENIUS_TOKEN", "")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

if not GENIUS_TOKEN:
    raise ValueError("Please set GENIUS_TOKEN in your .env file")
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise ValueError("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file")

OUTPUT_PATH = project_root / "data" / "processed" / "combined_songs.json"

def get_lyrics_from_url(url):
    """Scrape lyrics from a Genius song URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        if lyrics_divs:
            lyrics = '\n'.join([div.get_text(separator='\n') for div in lyrics_divs])
            lyrics = lyrics.strip()
            
            lines = lyrics.split('\n')
            cleaned_lines = []
            skip_mode = True
            
            for line in lines:
                if any(marker in line for marker in ['[Verse', '[Chorus', '[Bridge', '[Intro', '[Outro', '[Pre-', '[Refrain', '[Hook', '[Post-']):
                    skip_mode = False
                
                if not skip_mode:
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                return '\n'.join(cleaned_lines).strip()
            return lyrics
        return None
    except Exception as e:
        print(f"    Error scraping lyrics: {e}")
        return None

def get_spotify_client():
    """Initialize Spotify client."""
    credentials = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    return spotipy.Spotify(client_credentials_manager=credentials)

def normalize_title(title):
    """Normalize song title for matching."""
    title = title.lower()
    title = title.replace("'", "").replace("'", "")
    title = title.replace("  ", " ").strip()
    for suffix in [" - remaster", " - remastered", " (remaster)", " (remastered)", 
                   " - radio edit", " (radio edit)", " - single version"]:
        title = title.replace(suffix, "")
    return title

def similarity(a, b):
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()

def download_all(artists, songs_per_artist=10):
    """Download songs from both Spotify and Genius, then merge."""
    print("="*70)
    print("DOWNLOADING MUSIC DATA FROM SPOTIFY + GENIUS")
    print("="*70)
    
    sp = get_spotify_client()
    genius_headers = {'Authorization': f'Bearer {GENIUS_TOKEN}'}
    all_songs = []
    
    for artist_name in artists:
        print(f"\n{'='*70}")
        print(f"Processing: {artist_name}")
        print(f"{'='*70}")
        
        try:
            print(f"\n[1/2] Fetching from Spotify...")
            
            results = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
            if not results['artists']['items']:
                print(f"  ✗ Artist not found on Spotify")
                continue
            
            artist = results['artists']['items'][0]
            artist_id = artist['id']
            artist_spotify_name = artist['name']
            
            top_tracks = sp.artist_top_tracks(artist_id)
            tracks = top_tracks['tracks'][:songs_per_artist]
            
            print(f"  ✓ Found {len(tracks)} tracks on Spotify")
            print(f"\n[2/2] Fetching lyrics from Genius...")
            
            for track in tracks:
                song_data = {
                    "title": track['name'],
                    "artist": artist_name,
                    "spotify_artist_name": artist_spotify_name,
                    "album": track['album']['name'],
                    "release_date": track['album']['release_date'],
                    "duration_ms": track['duration_ms'],
                    "popularity": track['popularity']
                }
                
                    # Note: Audio features endpoint is currently blocked (403)
                    # Uncomment below when issue is resolved
                    # try:
                    #     audio_features = sp.audio_features([track['id']])[0]
                    #     if audio_features:
                    #         song_data.update({
                    #             "tempo": audio_features['tempo'],
                    #             "key": audio_features['key'],
                    #             ...
                    #         })
                    # except Exception as e:
                    #     pass
                
                # Find lyrics on Genius
                try:
                    search_query = f"{artist_name} {track['name']}"
                    search_url = f'https://api.genius.com/search?q={search_query}'
                    response = requests.get(search_url, headers=genius_headers)
                    response.raise_for_status()
                    
                    hits = response.json().get('response', {}).get('hits', [])
                    
                    best_match = None
                    best_score = 0
                    
                    for hit in hits[:5]:
                        result = hit['result']
                        score = similarity(track['name'], result.get('title', ''))
                        
                        if score > best_score and score >= 0.7:
                            best_score = score
                            best_match = result
                    
                    if best_match:
                        genius_url = best_match.get('url', '')
                        lyrics = get_lyrics_from_url(genius_url)
                        
                        if lyrics:
                            song_data['lyrics'] = lyrics
                            song_data['match_score'] = best_score
                            print(f"  ✓ {track['name'][:50]:50} [lyrics ✓]")
                        else:
                            # Skip songs with no lyrics
                            continue
                    else:
                        # Skip songs with no lyrics match
                        continue
                    
                    time.sleep(0.3)
                    
                except Exception as e:
                    # Skip on error fetching lyrics
                    print(f"  ✗ {track['name'][:50]:50} [error fetching lyrics]")
                    continue
                
                all_songs.append(song_data)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Error processing {artist_name}: {e}")
    
    # Since we only keep songs with lyrics, total_songs == songs_with_lyrics
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('w', encoding='utf-8') as f:
        json.dump({
            "source": "Spotify + Genius",
            "total_songs": len(all_songs),
            "items": all_songs
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETE!")
    print(f"{'='*70}")
    print(f"Total songs with lyrics: {len(all_songs)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    artists = [
        "Taylor Swift",
        "The Beatles",
        "Ed Sheeran",
        "Adele",
        "Drake"
    ]
    
    download_all(artists, songs_per_artist=20)
