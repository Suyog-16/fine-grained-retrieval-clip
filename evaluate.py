#!/usr/bin/env python3
"""
CLI script to evaluate Baseline CLIP vs Cross-Attention Reranker on the test set.
Usage:
    python evaluate.py
    python evaluate.py --checkpoint best_cross_attention_reranker.pt --pool_size 50 --alpha 0.5
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import clip

from src.models import CrossAttentionReranker
from src.dataset import DressImageDataset
from src.retrieval import TwoStageRetriever

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Grained Retrieval on Dress Test Set")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dress CSVs and images")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP backbone architecture")
    parser.add_argument("--checkpoint", type=str, default="notebooks/best_cross_attention_reranker.pt", help="Path to trained reranker weights")
    parser.add_argument("--pool_size", type=int, default=50, help="Candidate pool size (M) for Stage 2 reranking")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blending weight (1.0 = reranker only, 0.0 = CLIP only)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for gallery precomputation")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device}")

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "dress_images" / "test"
    test_csv = data_dir / "dress_test.csv"

    # 1. Load CLIP Model
    print(f"Loading CLIP {args.clip_model}...")
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()

    # 2. Load Reranker Model
    reranker = None
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Fallback to root or notebooks
        if Path("best_cross_attention_reranker.pt").exists():
            ckpt_path = Path("best_cross_attention_reranker.pt")

    if ckpt_path.exists():
        print(f"Loading reranker weights from {ckpt_path}...")
        reranker = CrossAttentionReranker().to(device)
        reranker.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
        reranker.eval()
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found. Evaluating Baseline CLIP only.")

    # 3. Load Test Data & Build Gallery
    test_df = pd.read_csv(test_csv)
    print(f"Loaded test split: {len(test_df):,} dress queries")

    test_ds = DressImageDataset(test_df, img_dir, preprocess)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    retriever = TwoStageRetriever(clip_model=model, reranker=reranker, device=device)
    retriever.build_gallery(test_loader, cache_patches=(reranker is not None))

    # 4. Run Benchmark Evaluation
    retriever.evaluate(
        df=test_df,
        candidate_pool_size=args.pool_size,
        alpha=args.alpha,
        split_name=f"Test Set, Pool M={args.pool_size}, alpha={args.alpha}"
    )

if __name__ == "__main__":
    main()
