"""Create/normalize Qwen-style labels for poem-song pairs.

Features:
- Read input candidate pairs from a JSONL file (or generate negatives by sampling songs).
- Score pairs using a local Hugging Face model (if requested).
- Parse numeric scores and normalize to 0..1.
- Optionally generate negatives per poem (random + hard sampling by index).

Usage examples:
  python src/utils/qwen_label_pairs.py --input data/processed/qwen_input.jsonl \
      --output data/processed/qwen_labels_out.jsonl --use-local-model --local-model-id gpt2 --device cpu

Notes:
- The script prefers a local HF model; remote/API code can be added where noted.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            json.dump(it, f, ensure_ascii=False)
            f.write("\n")


def load_songs(path: Path) -> List[Dict[str, Any]]:
    # Expected format: list of song dicts with at least `id` and `lyrics` or `text`.
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "songs" in data:
        return data["songs"]
    if isinstance(data, list):
        return data
    # Fallback: wrap into list
    return [data]


def make_prompt(poem_text: str, song_text: str) -> str:
    # Short, natural prompt asking for a single numeric score.
    return (
        "Rate how well the song lyrics match the poem on a scale from 0 (no match) to 10 (very good match). "
        "Give only a single numeric score, and nothing else.\n\nPoem:\n" + poem_text + "\n\nSong:\n" + song_text + "\n\nScore:" 
    )


def parse_response(resp: str) -> Dict[str, Any]:
    """Extract a numeric score if present and normalize to 0..1.

    Returns a dict with `raw` (float or None) and `score` (0..1 or None).
    """
    # Find a number like 7 or 7.5 or 0.8
    m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", resp)
    if not m:
        return {"raw": None, "score": None, "text": resp}
    raw = float(m.group(1))
    # Heuristic: if raw <= 1.0 treat as 0..1 else 0..10
    if 0.0 <= raw <= 1.0:
        score = raw
    else:
        # clamp and normalize
        raw_clamped = max(0.0, min(raw, 10.0))
        score = raw_clamped / 10.0
    return {"raw": raw, "score": score, "text": resp}


def safe_generate_local(model, tokenizer, prompt: str, device: str = "cpu", max_new_tokens: int = 20) -> str:
    # Lazy import to keep script import-light until needed.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover - informative
        raise RuntimeError("Missing transformers/torch. Install via pip install transformers torch") from e

    # If `model` is a str id, load; otherwise assume tuple(tokenizer, model)
    if isinstance(model, str):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return trailing text after the prompt, if present
    if prompt in text:
        return text[text.index(prompt) + len(prompt) :].strip()
    return text.strip()


def label_pairs(
    pairs: Iterable[Dict[str, Any]],
    output_path: Path,
    use_local_model: bool = False,
    local_model_id: Optional[str] = None,
    device: str = "cpu",
    max_new_tokens: int = 20,
):
    # If using a local model, prepare tokenizer/model id; keep lazy-loading.
    model_obj = None
    if use_local_model:
        if not local_model_id:
            raise ValueError("--local-model-id required when --use-local-model is set")
        model_obj = local_model_id
    out_items: List[Dict[str, Any]] = []
    for i, p in enumerate(pairs):
        poem = p.get("poem_text") or p.get("poem") or p.get("poem_text_raw") or p.get("poem_body") or ""
        song = p.get("song_text") or p.get("song") or p.get("song_text_raw") or p.get("lyrics") or ""
        prompt = make_prompt(poem, song)
        if use_local_model:
            resp_text = safe_generate_local(model_obj, None, prompt, device=device, max_new_tokens=max_new_tokens)
        else:
            # Placeholder for remote API call. Keep wording simple and neutral.
            raise RuntimeError("Remote API not implemented. Use --use-local-model for local generation.")
        parsed = parse_response(resp_text)
        out = {
            "poem_id": p.get("poem_id"),
            "song_id": p.get("song_id"),
            "poem_text": poem,
            "song_text": song,
            "raw_score": parsed["raw"],
            "score": parsed["score"],
            "response_text": parsed["text"],
        }
        out_items.append(out)
    write_jsonl(output_path, out_items)


def generate_negatives(
    poems: List[Dict[str, Any]],
    songs: List[Dict[str, Any]],
    n_random: int = 2,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    out: List[Dict[str, Any]] = []
    song_pool = songs
    for poem in poems:
        poem_text = poem.get("poem_text") or poem.get("text") or poem.get("body") or ""
        poem_id = poem.get("id") or poem.get("poem_id")
        # sample random songs as negatives
        negs = rnd.sample(song_pool, k=min(n_random, len(song_pool)))
        for s in negs:
            out.append(
                {
                    "poem_id": poem_id,
                    "song_id": s.get("id") or s.get("song_id"),
                    "poem_text": poem_text,
                    "song_text": s.get("lyrics") or s.get("text") or "",
                    "generated_negative": True,
                }
            )
    return out


def main():
    p = argparse.ArgumentParser(description="Label poem-song pairs with numeric scores (0..1 normalized).")
    p.add_argument("--input", type=Path, help="Input candidate JSONL (one pair per line).")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL with scores.")
    p.add_argument("--use-local-model", action="store_true", help="Use a local HF model for scoring.")
    p.add_argument("--local-model-id", type=str, help="HF model id (e.g., gpt2, or a larger model).")
    p.add_argument("--device", type=str, default="cpu", help="Device for model ('cpu' or 'cuda').")
    p.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate for a score (short).")
    p.add_argument("--generate-negatives", action="store_true", help="Generate random negatives using songs file.")
    p.add_argument("--songs-file", type=Path, help="JSON file with songs (used for negative generation).")
    p.add_argument("--poems-file", type=Path, help="JSON file with poems (used for negative generation).")
    p.add_argument("--neg-random-n", type=int, default=2, help="Random negatives per poem when generating negatives.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible negatives.")
    args = p.parse_args()

    if args.generate_negatives:
        if not args.songs_file or not args.poems_file:
            raise ValueError("--songs-file and --poems-file required when --generate-negatives is set")
        songs = load_songs(args.songs_file)
        poems = load_songs(args.poems_file)  # poem file can be similar shape
        negs = generate_negatives(poems, songs, n_random=args.neg_random_n, seed=args.seed)
        # If an input JSONL is provided, append negatives to it and label all
        base_pairs: List[Dict[str, Any]] = []
        if args.input:
            base_pairs = load_jsonl(args.input)
        base_pairs.extend(negs)
        label_pairs(base_pairs, args.output, use_local_model=args.use_local_model, local_model_id=args.local_model_id, device=args.device, max_new_tokens=args.max_new_tokens)
        print(f"Wrote {len(base_pairs)} pairs (including negatives) to {args.output}")
        return

    # Normal labeling path
    if not args.input:
        raise ValueError("--input is required unless --generate-negatives is used")
    pairs = load_jsonl(args.input)
    label_pairs(pairs, args.output, use_local_model=args.use_local_model, local_model_id=args.local_model_id, device=args.device, max_new_tokens=args.max_new_tokens)
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
