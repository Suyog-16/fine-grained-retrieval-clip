# ── 0. install ───────────────────────────────────────
# pip install datasets pandas pillow

from datasets import load_dataset
import pandas as pd
import random
from collections import defaultdict

random.seed(42)

# ── 1. load ──────────────────────────────────────────
# streams from HuggingFace, no manual download needed
ds = load_dataset("Marqo/fashion200k", split="data")
print(f"Total rows: {len(ds)}")          # ~202k
print(ds.column_names)
# ['image', 'category1', 'category2', 'category3', 'text', 'item_ID']

# ── 2. filter to dresses only ─────────────────────────
dress_ds = ds.filter(lambda x: x["category1"] == "dresses")
print(f"Dress rows: {len(dress_ds)}")    # ~80-90k (multiple views per product)

# ── 3. deduplicate to one view per product ────────────
# item_ID format is "productid_viewnumber" e.g. "51727804_0"
# keep only _0 (first/canonical view) to avoid leaking
# the same dress into both train and test
seen_products = set()
canonical = []

for item in dress_ds:
    product_id = item["item_ID"].rsplit("_", 1)[0]
    view_num   = item["item_ID"].rsplit("_", 1)[1]
    if view_num == "0" and product_id not in seen_products:
        seen_products.add(product_id)
        canonical.append(item)

print(f"Unique dress products: {len(canonical)}")   # ~15-20k

# ── 4. stratified sample by subcategory ──────────────
N_TOTAL   = 15000
N_TRAIN   = 12000
N_VAL     =  1500
N_TEST    =  1500

buckets = defaultdict(list)
for item in canonical:
    buckets[item["category2"]].append(item)

print("Subcategories:")
for k, v in sorted(buckets.items(), key=lambda x: -len(x[1])):
    print(f"  {k}: {len(v)}")

# proportional sample from each subcategory
sampled = []
total   = len(canonical)
for bucket_items in buckets.values():
    k = max(1, int(N_TOTAL * len(bucket_items) / total))
    sampled.extend(random.sample(bucket_items, min(k, len(bucket_items))))

random.shuffle(sampled)
sampled = sampled[:N_TOTAL]

train = sampled[:N_TRAIN]
val   = sampled[N_TRAIN : N_TRAIN + N_VAL]
test  = sampled[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# ── 5. save CSVs (paths + captions, no images yet) ───
def to_df(items):
    return pd.DataFrame([{
        "item_ID":   x["item_ID"],
        "category2": x["category2"],
        "category3": x["category3"],   # short title
        "text":      x["text"],         # long description
    } for x in items])

to_df(train).to_csv("dress_train.csv", index=False)
to_df(val).to_csv("dress_val.csv",     index=False)
to_df(test).to_csv("dress_test.csv",   index=False)
print("CSVs saved.")

# ── 6. save images to disk ────────────────────────────
import os
from pathlib import Path

IMG_DIR = Path("dress_images")
IMG_DIR.mkdir(exist_ok=True)

for split_name, items in [("train", train), ("val", val), ("test", test)]:
    split_dir = IMG_DIR / split_name
    split_dir.mkdir(exist_ok=True)
    for item in items:
        img_path = split_dir / f"{item['item_ID']}.jpg"
        if not img_path.exists():
            item["image"].save(img_path)   # PIL Image object from HF

print("Images saved.")