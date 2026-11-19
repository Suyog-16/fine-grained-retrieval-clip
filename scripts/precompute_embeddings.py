import os
import argparse
import clip
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def get_patch_embeddings(visual_model, x):
    """
    Extracts all patch embeddings from CLIP's ViT, not just the CLS token.
    Applies the final layer norm and projection to all tokens.
    """
    x = visual_model.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([visual_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + visual_model.positional_embedding.to(x.dtype)
    x = visual_model.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = visual_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = visual_model.ln_post(x)

    if visual_model.proj is not None:
        x = x @ visual_model.proj

    return x

def precompute_split(csv_path, images_dir, output_dir, model, preprocess, device, batch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # We will process in batches to be efficient
    paths = []
    item_ids = []
    
    for idx, row in df.iterrows():
        item_id = row['item_ID']
        img_path = Path(images_dir) / f"{item_id}.jpg"
        if img_path.exists():
            paths.append(img_path)
            item_ids.append(item_id)
        else:
            print(f"Warning: Image missing {img_path}")
            
    print(f"Processing {len(paths)} images for {csv_path}...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size)):
            batch_paths = paths[i:i+batch_size]
            batch_ids = item_ids[i:i+batch_size]
            
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
                
            image_tensor = torch.stack(images).to(device)
            
            # Forward pass through modified vision model
            patch_embeds = get_patch_embeddings(model.visual, image_tensor)
            patch_embeds = patch_embeds.cpu() # shape: [B, N, D]
            
            # Save individual files
            for j, item_id in enumerate(batch_ids):
                save_path = Path(output_dir) / f"{item_id}.pt"
                torch.save(patch_embeds[j].clone(), save_path)

def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP patch embeddings")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with CSVs and images")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    
    # Precompute for train, val, test
    splits = ["train", "val", "test"]
    for split in splits:
        csv_path = Path(args.data_dir) / f"dress_{split}.csv"
        images_dir = Path(args.data_dir) / "dress_images" / split
        output_dir = Path(args.data_dir) / "patch_embeddings" / split
        
        if csv_path.exists():
            precompute_split(csv_path, images_dir, output_dir, model, preprocess, device, args.batch_size)

if __name__ == "__main__":
    main()
