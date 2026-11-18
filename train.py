#!/usr/bin/env python3
"""
CLI Script to train the Cross-Attention Reranker with InfoNCE loss.
Example:
    python train.py --epochs 3 --batch_size 16 --lr 1e-4
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip

from src.models import CrossAttentionReranker, extract_text_tokens, extract_patch_tokens
from src.dataset import DressPairDataset, DressImageDataset
from src.retrieval import TwoStageRetriever

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Cross-Attention Reranker on Fashion Pairs")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory containing CSVs and images")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model architecture")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE softmax temperature")
    parser.add_argument("--loss", type=str, default="infonce", choices=["infonce", "bce"],
                        help="Loss function: infonce (default) or bce")
    parser.add_argument("--n_train_samples", type=int, default=None, help="Optional subset of training pairs")
    parser.add_argument("--n_val_samples", type=int, default=None, help="Optional subset of validation pairs")
    parser.add_argument("--output_path", type=str, default="best_cross_attention_reranker.pt", help="Path to save weights")
    parser.add_argument("--eval_after_training", action="store_true", help="Run benchmark on test set after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def train():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "dress_images"

    # 1. Load CLIP backbone (frozen)
    print(f"Loading CLIP {args.clip_model}...")
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()

    # 2. Load Data Splits
    train_df = pd.read_csv(data_dir / "dress_train.csv")
    val_df   = pd.read_csv(data_dir / "dress_val.csv")

    if args.n_train_samples is not None:
        train_df = train_df.iloc[:args.n_train_samples]
    if args.n_val_samples is not None:
        val_df = val_df.iloc[:args.n_val_samples]

    print(f"Training on {len(train_df):,} pairs | Validating on {len(val_df):,} pairs")

    train_ds = DressPairDataset(train_df, img_dir / "train", preprocess)
    val_ds   = DressPairDataset(val_df, img_dir / "val", preprocess)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)

    # 3. Model & Optimizer
    reranker = CrossAttentionReranker().to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_val_loss = float("inf")

    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs ({args.loss.upper()} loss)...")
    for epoch in range(1, args.epochs + 1):
        reranker.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for images, texts, _ in pbar:
            with torch.no_grad():
                tokens = clip.tokenize(list(texts), truncate=True).to(device)
                text_toks = extract_text_tokens(model, tokens)
                patch_toks = extract_patch_tokens(model, images.to(device))

            B = len(texts)
            # Expand to compute all B x B pairs in batch
            t_exp = text_toks.unsqueeze(1).expand(B, B, 77, 512).reshape(B * B, 77, 512)
            p_exp = patch_toks.unsqueeze(0).expand(B, B, 50, 512).reshape(B * B, 50, 512)

            scores = reranker(t_exp, p_exp).reshape(B, B)
            if args.loss == "infonce":
                labels = torch.arange(B, device=device)
                loss = F.cross_entropy(scores / args.temperature, labels)
            else:  # bce
                labels = torch.eye(B, device=device)
                loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss / len(train_loader)

        # Validation
        reranker.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts, _ in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]", leave=False):
                tokens = clip.tokenize(list(texts), truncate=True).to(device)
                text_toks = extract_text_tokens(model, tokens)
                patch_toks = extract_patch_tokens(model, images.to(device))

                B = len(texts)
                t_exp = text_toks.unsqueeze(1).expand(B, B, 77, 512).reshape(B * B, 77, 512)
                p_exp = patch_toks.unsqueeze(0).expand(B, B, 50, 512).reshape(B * B, 50, 512)

                scores = reranker(t_exp, p_exp).reshape(B, B)
                if args.loss == "infonce":
                    labels = torch.arange(B, device=device)
                    loss = F.cross_entropy(scores / args.temperature, labels)
                else:  # bce
                    labels = torch.eye(B, device=device)
                    loss = F.binary_cross_entropy_with_logits(scores, labels)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch:02d} -> Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(reranker.state_dict(), args.output_path)
            print(f"  [+] Saved new best model to {args.output_path} (val_loss: {best_val_loss:.4f})")

    print(f"\nTraining finished! Best model saved to: {args.output_path}")

    # 5. Optional Evaluation
    if args.eval_after_training:
        print("\nEvaluating trained reranker on test set...")
        reranker.load_state_dict(torch.load(args.output_path, weights_only=True, map_location=device))
        test_df = pd.read_csv(data_dir / "dress_test.csv")
        test_ds = DressImageDataset(test_df, img_dir / "test", preprocess)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

        retriever = TwoStageRetriever(clip_model=model, reranker=reranker, device=device)
        retriever.build_gallery(test_loader, cache_patches=True)
        retriever.evaluate(test_df, candidate_pool_size=50, alpha=0.5, split_name="Test Set")

if __name__ == "__main__":
    train()
