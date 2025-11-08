# The Sound of a Sonnet: Finding Musical Counterparts for Poetry with Unsupervised Learning

**CS 229 Final Project**

**Team Members:** Shirley Zhang, Cheney Sang, Amelia Sarah Bloom

## Project Overview
Creating a model to recommend songs that match the mood of poems using Sentence-BERT for text encoding and unsupervised learning to align poem and song lyric embeddings.

## Project Structure

```
.
├── data/                      # Data directory
│   ├── raw/                   # Raw poems and song lyrics
│   └── processed/             # Preprocessed and cleaned data
│
├── notebooks/                 # Jupyter notebooks for exploration and analysis
│
├── src/                       # Source code
│   ├── data/                  # Data collection and preprocessing scripts
│   ├── models/                # Model implementations (Sentence-BERT encoder, clustering)
│   └── utils/                 # Utility functions (similarity metrics, visualization)
│
├── results/                   # Results and outputs
│   ├── figures/               # Plots and visualizations
│   └── embeddings/            # Saved embeddings
│
├── tests/                     # Unit tests
│
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment specification
└── README.md                  # This file
```

## Setup

### Using pip:
```bash
pip install -r requirements.txt
```

### Using conda:
```bash
conda env create -f environment.yml
conda activate cs229
```

## Quick local run

Collect songs with lyrics locally (uses your .env in the project root):
```bash
python src/data/download_bulk_songs.py \
	--artists-file data/artists.txt \
	--target 3000 --songs-per-artist 40 \
	--output data/processed/combined_songs_large.json
```

## Methods
- **Text Encoder:** Sentence-BERT to vectorize poems and song lyrics
- **Unsupervised Learning:** Create labels to compare moods of poems and songs
- **Evaluation:** Cosine similarity, clustering metrics (Silhouette, Davies-Bouldin), modality-based validation

## Experiments
- 80/20 train-test split with contrastive alignment
- Clustering analysis (k-means) for mood classification
- t-SNE/UMAP visualization of embedding spaces
- User survey for human-perceived relevance validation
