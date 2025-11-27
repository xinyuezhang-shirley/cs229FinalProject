"""
Train a poem↔song projection model with hard negatives, margin loss, and optional human labels.

Usage (defaults use existing embeddings/features):
    python -m src.model.train_projection --epochs 10 --batch-size 128

Optional human labels:
    - Provide a JSONL file with objects like:
        {"poem_index": 42, "song_index_aligned": 17, "label": 1}
      where poem_index references data/raw/poetrydb_poems.json (0-based)
      and song_index_aligned references the filtered/reordered songs used in
      data/processed/additional_features.npz (0-based).
    - You can also supply "song_index_raw" (0-based in combined_songs_large_fixed.json);
      it will be mapped to the aligned index using songs_source_indexes.
    - For convenience, "song_title" + "song_artist" may be used and will be matched
      case-insensitively to the raw songs list (best-effort, first match wins).
    Place the file at data/processed/human_labels.jsonl or pass --human-labels PATH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_embeddings(
    poems_path: Path,
    songs_path: Path,
    features_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poem_vecs = np.load(poems_path)
    song_vecs = np.load(songs_path)

    feats = np.load(features_path)
    poem_feats = feats["poems_standardized"]
    song_feats = feats["songs_standardized"]
    song_source_indexes = feats["songs_source_indexes"]

    # Align songs to the feature file order (some songs were filtered out)
    song_vecs = song_vecs[song_source_indexes]

    poem_in = np.concatenate([poem_vecs, poem_feats], axis=1)
    song_in = np.concatenate([song_vecs, song_feats], axis=1)

    return poem_in, song_in, poem_vecs, song_vecs, song_source_indexes


def compute_seed_pairs(
    poem_vecs: np.ndarray,
    song_vecs: np.ndarray,
    pos_pct: float,
    neg_pct: float,
    hard_low_pct: float,
    hard_high_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poems_norm = poem_vecs / np.linalg.norm(poem_vecs, axis=1, keepdims=True)
    songs_norm = song_vecs / np.linalg.norm(song_vecs, axis=1, keepdims=True)
    cos_matrix = poems_norm @ songs_norm.T
    cos_vals = cos_matrix.ravel()

    neg_thresh = np.percentile(cos_vals, neg_pct)
    hard_low = np.percentile(cos_vals, hard_low_pct)
    hard_high = np.percentile(cos_vals, hard_high_pct)
    pos_thresh = np.percentile(cos_vals, pos_pct)

    pos_pairs = np.argwhere(cos_matrix >= pos_thresh)
    neg_pairs = np.argwhere(cos_matrix <= neg_thresh)
    hard_pairs = np.argwhere((cos_matrix >= hard_low) & (cos_matrix <= hard_high))
    return cos_matrix, pos_pairs, neg_pairs, hard_pairs


def load_human_labels(
    path: Path,
    song_source_indexes: np.ndarray,
    raw_songs: list[dict],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (positives, negatives) as lists of (poem_idx, aligned_song_idx)."""
    positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int]] = []

    if not path.exists():
        return positives, negatives

    # Map raw song index -> aligned song index
    raw_to_aligned = {int(raw_idx): i for i, raw_idx in enumerate(song_source_indexes)}

    # Simple title/artist lookup
    title_artist_to_raw = {}
    for idx, song in enumerate(raw_songs):
        key = (song.get("title", "").lower().strip(), song.get("artist", "").lower().strip())
        title_artist_to_raw[key] = idx

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            poem_idx = int(entry["poem_index"])
            aligned_idx = None

            if "song_index_aligned" in entry:
                aligned_idx = int(entry["song_index_aligned"])
            elif "song_index_raw" in entry:
                raw_idx = int(entry["song_index_raw"])
                aligned_idx = raw_to_aligned.get(raw_idx)
            elif "song_title" in entry and "song_artist" in entry:
                key = (entry["song_title"].lower().strip(), entry["song_artist"].lower().strip())
                raw_idx = title_artist_to_raw.get(key)
                if raw_idx is not None:
                    aligned_idx = raw_to_aligned.get(raw_idx)

            if aligned_idx is None:
                continue

            label = int(entry.get("label", 1))
            if label == 1:
                positives.append((poem_idx, aligned_idx))
            else:
                negatives.append((poem_idx, aligned_idx))

    return positives, negatives


class PairDataset(Dataset):
    def __init__(
        self,
        poem_in: np.ndarray,
        song_in: np.ndarray,
        pos_pairs: np.ndarray,
        neg_pairs: np.ndarray,
        hard_pairs: np.ndarray,
        human_pos: list[tuple[int, int]],
        human_neg: list[tuple[int, int]],
        size: int = 50000,
    ) -> None:
        self.poem_in = poem_in
        self.song_in = song_in
        if human_pos:
            self.pos_pairs = np.concatenate([pos_pairs, np.array(human_pos, dtype=np.int64)], axis=0)
        else:
            self.pos_pairs = pos_pairs

        if human_neg:
            self.neg_pairs = np.concatenate([neg_pairs, np.array(human_neg, dtype=np.int64)], axis=0)
        else:
            self.neg_pairs = neg_pairs
        self.hard_pairs = hard_pairs
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        i, j_pos = self.pos_pairs[np.random.randint(len(self.pos_pairs))]

        # 50% hard negatives, otherwise easy negatives
        if len(self.hard_pairs) > 0 and np.random.rand() < 0.5:
            i_neg, j_neg = self.hard_pairs[np.random.randint(len(self.hard_pairs))]
        else:
            i_neg, j_neg = self.neg_pairs[np.random.randint(len(self.neg_pairs))]

        p = torch.tensor(self.poem_in[i], dtype=torch.float32)
        s_pos = torch.tensor(self.song_in[j_pos], dtype=torch.float32)
        s_neg = torch.tensor(self.song_in[j_neg], dtype=torch.float32)
        return p, s_pos, s_neg


class ProjectionModel(nn.Module):
    def __init__(self, p_dim: int, s_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.poem_proj = nn.Sequential(
            nn.Linear(p_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )
        self.song_proj = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, p: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        p_z = self.poem_proj(p)
        s_z = self.song_proj(s)
        p_norm = F.normalize(p_z, dim=1)
        s_norm = F.normalize(s_z, dim=1)
        return (p_norm * s_norm).sum(dim=1)


def clip_loss(p_z: torch.Tensor, s_z: torch.Tensor, temperature: float) -> torch.Tensor:
    p = F.normalize(p_z, dim=1)
    s = F.normalize(s_z, dim=1)
    logits = (p @ s.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_p_to_s = F.cross_entropy(logits, labels)
    loss_s_to_p = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_p_to_s + loss_s_to_p)


def triplet_margin(
    p_z: torch.Tensor,
    s_pos_z: torch.Tensor,
    s_neg_z: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    p = F.normalize(p_z, dim=1)
    s_pos = F.normalize(s_pos_z, dim=1)
    s_neg = F.normalize(s_neg_z, dim=1)
    pos_sim = (p * s_pos).sum(dim=1)
    neg_sim = (p * s_neg).sum(dim=1)
    return F.relu(pos_sim - neg_sim + margin).mean()


def train(args: argparse.Namespace) -> None:
    poem_in, song_in, poem_vecs, song_vecs, song_source_indexes = load_embeddings(
        PROJECT_ROOT / "data" / "processed" / "mpnet_embeddings_poems.npy",
        PROJECT_ROOT / "data" / "processed" / "mpnet_embeddings_songs.npy",
        PROJECT_ROOT / "data" / "processed" / "additional_features.npz",
    )

    with (PROJECT_ROOT / "data" / "processed" / "combined_songs_large_fixed.json").open(
        "r", encoding="utf-8"
    ) as f:
        raw_songs = json.load(f)["items"]

    cos_matrix, pos_pairs, neg_pairs, hard_pairs = compute_seed_pairs(
        poem_vecs,
        song_vecs,
        pos_pct=args.pos_pct,
        neg_pct=args.neg_pct,
        hard_low_pct=args.hard_low_pct,
        hard_high_pct=args.hard_high_pct,
    )

    human_pos, human_neg = load_human_labels(
        args.human_labels,
        song_source_indexes,
        raw_songs,
    )

    dataset = PairDataset(
        poem_in,
        song_in,
        pos_pairs,
        neg_pairs,
        hard_pairs,
        human_pos,
        human_neg,
        size=args.samples_per_epoch,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectionModel(poem_in.shape[1], song_in.shape[1], args.proj_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for p, s_pos, s_neg in loader:
            p = p.to(device)
            s_pos = s_pos.to(device)
            s_neg = s_neg.to(device)

            p_z = model.poem_proj(p)
            s_pos_z = model.song_proj(s_pos)
            s_neg_z = model.song_proj(s_neg)

            loss_clip = clip_loss(p_z, s_pos_z, args.temperature)
            loss_triplet = triplet_margin(p_z, s_pos_z, s_neg_z, args.margin)
            loss_align = F.mse_loss(p_z, s_pos_z)

            loss = (
                args.contrast_weight * loss_clip
                + args.triplet_weight * loss_triplet
                + args.align_weight * loss_align
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg = total / len(loader)
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "proj_dim": args.proj_dim,
            "temperature": args.temperature,
        },
        args.output,
    )
    print(f"Saved model to {args.output}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train projection model with hard negatives and margin loss.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--align-weight", type=float, default=0.5)
    parser.add_argument("--pos-pct", type=float, default=90.0)
    parser.add_argument("--neg-pct", type=float, default=20.0)
    parser.add_argument("--hard-low-pct", type=float, default=40.0)
    parser.add_argument("--hard-high-pct", type=float, default=70.0)
    parser.add_argument(
        "--human-labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "human_labels.jsonl",
        help="Optional JSONL with human labels (see file docstring for schema).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "projection_hardneg.pt",
        help="Where to save the trained model checkpoint.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (use 0 on macOS if shm is restricted).")
    parser.add_argument("--pin-memory", action="store_true", help="Use pinned memory in DataLoader (GPU only).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(args)
