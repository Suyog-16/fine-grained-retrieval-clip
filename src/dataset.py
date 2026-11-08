import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import clip

class DressImageDataset(Dataset):
    """Dataset for gallery evaluation: returns preprocessed image and item_ID."""
    def __init__(self, df, img_dir, preprocess):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['item_ID']}.jpg"
        image = Image.open(img_path).convert("RGB")
        return self.preprocess(image), str(row["item_ID"])

class DressPairDataset(Dataset):
    """Dataset for reranker training: returns image, query text, and item_ID."""
    def __init__(self, df, img_dir, preprocess):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['item_ID']}.jpg"
        image = Image.open(img_path).convert("RGB")
        return self.preprocess(image), str(row["text"]), str(row["item_ID"])

class CrossModalDataset(Dataset):
    def __init__(self, csv_path, patch_dir):
        """
        Args:
            csv_path: Path to the split CSV (e.g., dress_train.csv)
            patch_dir: Directory containing precomputed .pt patch embeddings
        """
        self.df = pd.read_csv(csv_path)
        self.patch_dir = Path(patch_dir)
        
        # Filter dataframe to only include items with precomputed patches
        valid_items = []
        for idx, row in self.df.iterrows():
            item_id = row['item_ID']
            if (self.patch_dir / f"{item_id}.pt").exists():
                valid_items.append(row)
        
        self.df = pd.DataFrame(valid_items)
        print(f"Loaded {len(self.df)} valid items from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_id = row['item_ID']
        text = row['text']
        
        # Load precomputed patch embeddings
        patch_path = self.patch_dir / f"{item_id}.pt"
        patch_embeds = torch.load(patch_path, weights_only=True)  # Shape: [num_patches, embed_dim]
        
        # Tokenize text on the fly
        # clip.tokenize returns [1, 77], so we squeeze out the batch dimension
        tokenized_text = clip.tokenize(text, truncate=True).squeeze(0)
        
        return patch_embeds, tokenized_text, item_id

