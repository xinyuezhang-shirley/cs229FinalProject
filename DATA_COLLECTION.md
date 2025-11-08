# Data Collection Setup

This guide will help you set up both Spotify and Genius APIs to collect comprehensive music data.

## What You'll Get

- **From Spotify**: Audio features (tempo, key, energy, danceability, etc.), duration, popularity
- **From Genius**: Song lyrics
- **Combined**: Complete dataset with both audio features and lyrics

## Setup Instructions

### 1. Spotify API Setup

1. Go to https://developer.spotify.com/dashboard
2. Log in with your Spotify account (or create one)
3. Click "Create an App"
4. Fill in:
   - App name: "CS229 Music Analysis" (or any name)
   - App description: "For academic research"
   - Check the box agreeing to terms
5. Click "Create"
6. You'll see your **Client ID** and **Client Secret**
7. Copy both values to your `.env` file:
   ```
   SPOTIFY_CLIENT_ID=your_actual_client_id
   SPOTIFY_CLIENT_SECRET=your_actual_client_secret
   ```

### 2. Run Data Collection

Once your `.env` file is configured with both Genius and Spotify credentials:

```bash
# 1. Download Spotify data (audio features + metadata)
python src/data/download_spotify.py

# 2. Download Genius data (lyrics) - you already have this
python src/data/download_genius.py

# 3. Merge both datasets
python src/data/merge_data.py
```

### 3. Output Files

- `data/raw/spotify_songs.json` - Spotify data with audio features
- `data/raw/genius_songs.json` - Genius data with lyrics
- `data/processed/combined_songs.json` - **Merged dataset** (use this for your analysis!)

## Data Fields in Combined Dataset

Each song in the combined dataset includes:

**From Spotify:**
- `title`, `artist`, `album`
- `duration_ms`, `popularity`
- `tempo`, `key`, `mode`, `time_signature`
- `acousticness`, `danceability`, `energy`
- `instrumentalness`, `liveness`, `loudness`
- `speechiness`, `valence` (mood)
- `spotify_url`, `preview_url`

**From Genius:**
- `lyrics` (full text)
- `genius_url`
- `match_score` (confidence of the match)

## Troubleshooting

**"Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET"**
- Make sure you've added both credentials to `.env` file
- Remove quotes around the values
- Make sure there are no spaces around the `=` sign

**Low match rate when merging**
- Song titles might differ slightly between platforms
- The merge script uses fuzzy matching (80% similarity threshold)
- You can adjust the threshold in `merge_data.py`

## Next Steps

After collecting data, you can:
1. Analyze audio features vs. lyrics
2. Train models to predict song characteristics from lyrics
3. Cluster songs based on audio + lyrical features
4. Explore correlations between musical and lyrical properties
