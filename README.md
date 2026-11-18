# Fine-Grained Dress Retrieval Using CLIP & Cross-Attention Reranking

This repository explores **fine-grained text-to-image retrieval** on fashion items (specifically dresses) by combining **OpenAI CLIP (`ViT-B/32`)** with a custom **Cross-Attention Reranker** to improve its fine-grained image retrieval abilty.

Standard vision-language models like CLIP compress an entire image into a single 512-dimensional vector. While this effectively separates broad categories (e.g., distinguishing a "red dress" from a "black dress"), it struggles with subtle fashion attributes like sleeve length, lace textures, neckline cuts, or ruffle patterns. This project implements a **two-stage retrieval pipeline** that performs coarse semantic retrieval through vanilla CLIP followed by fine-grained token-to-patch attention reranking.

---

## System Architecture

![System Architecture](Architecture.png)

### The Two-Stage Retrieval Pipeline

To balance **retrieval speed** with **fine-grained precision**, we employ a two-stage approach:

- **Stage 1 — Fast Coarse Candidate Retrieval (Global CLIP):**
  - The entire test gallery of candidate dress images is pre-encoded into normalized 512-d CLIP vectors.
  - For any text query $Q$, dot-product similarity instantly extracts the **Top-$M$ candidates** (e.g., $M = 50$) in milliseconds.

- **Stage 2 — Fine-Grained Cross-Attention Reranking:**
  - For the top-$M$ candidate dresses, we extract contextualized text tokens ($77 \times 512$) from the CLIP text transformer and spatial patch tokens ($50 \times 512$: 1 `[CLS]` token + 49 spatial patches) from the ViT visual backbone.
  - A **Multi-Head Cross-Attention module** ($Q = \text{text tokens}, K/V = \text{visual patch tokens}$) allows specific textual descriptors (_"long sleeves"_, _"ruffled collar"_, _"lace overlay"_) to attend directly to their corresponding physical regions on the dress.
  - The attended output passes through a residual connection, LayerNorm, sequence mean-pooling, and an MLP to output a scalar relevance score $s_{\text{rerank}}$.
  - Scores are blended with coarse similarity ($\alpha = 0.5$):
    $$\text{Final Score} = \alpha \cdot s_{\text{rerank}} + (1 - \alpha) \cdot s_{\text{clip}}$$

---

## Detailed Evaluation & Benchmark Results

### 1. Evaluation Protocol

The benchmark evaluates all **1,197 dress queries** in `dress_test.csv` against the entire **1,197-image test gallery**:

1. For each test query, we compute similarity scores across all 1,197 candidate images.
2. We identify the exact 1-indexed rank of the ground-truth image (`item_ID`).
3. We record:
   - **Recall@K (R@K)**: Percentage of queries where the correct dress appears in the top $K$ positions ($K \in \{1, 5, 10\}$).
   - **Median Rank**: Median position of the true dress across all queries (lower is better).
   - **Mean Rank**: Average position of the true dress across all queries (lower is better).

### 2. Full Test Set Benchmark ($n = 1,197$)

| Metric          | Baseline CLIP (`ViT-B/32`) | Cross-Attention Reranker (Ours) |  **Improvement ($\Delta$)**   |
| :-------------- | :------------------------: | :-----------------------------: | :---------------------------: |
| **Recall@1**    |           11.70%           |           **12.53%**            | **+0.84%** (~+7.1% relative)  |
| **Recall@5**    |           29.82%           |           **32.08%**            | **+2.26%** (~+7.6% relative)  |
| **Recall@10**   |           42.11%           |           **45.36%**            | **+3.26%** (~+7.7% relative)  |
| **Median Rank** |             15             |             **13**              |  **-2 positions** (improved)  |
| **Mean Rank**   |            49.5            |            **48.1**             | **-1.4 positions** (improved) |

> **Key Takeaway:** The Cross-Attention Reranker achieves substantial gains in **R@5 (+2.26%)** and **R@10 (+3.26%)**, successfully pulling relevant dresses from ranks 10–25 up into the top recommendations.

### 3. Ablation: Candidate Pool Size ($M$)

## Dataset & Preparation

Built from **[Fashion200k](https://github.com/xthan/fashion-200k)**:

- Filtered strictly to `category1 == "dresses"`. to keep it simple
- Deduplicated to keep only the canonical first view (`_0`) per product, preventing identical dresses from leaking across splits.
- **Train split:** 12,000 dress items (`data/dress_train.csv`)
- **Validation split:** 1,500 dress items (`data/dress_val.csv`)
- **Test split:** 1,197 dress items (`data/dress_test.csv`)

To recreate the dataset from Hugging Face:

```bash
python scripts/download_dataset.py
```

---

## Project Structure

```
fine-grained-retrieval-clip/
├── src/                        # Core library (models, data, metrics, retrieval)
│   ├── __init__.py
│   ├── models.py               # CrossAttentionReranker + GlobalMLPReranker + feature extractors
│   ├── dataset.py              # DressImageDataset, DressPairDataset, CrossModalDataset
│   ├── metrics.py              # Recall@K, Median/Mean Rank computation
│   └── retrieval.py            # TwoStageRetriever (CLIP + reranker pipeline)
├── scripts/                    # Standalone CLI scripts (not importable modules)
│   ├── download_dataset.py     # Download & prepare Fashion200k dress subset from HuggingFace
│   └── precompute_embeddings.py # Precompute CLIP patch embeddings for all splits
├── notebooks/                  # Exploration and experiment notebooks
│   ├── data_exploration.ipynb
│   ├── experiment.ipynb
│   └── cross_attention_reranker_experiment.ipynb
├── checkpoints/                # Saved model weights (not committed to git)
│   ├── best_cross_attention_reranker.pt
│   └── best_global_mlp_reranker.pt
├── data/                       # Dataset files (CSVs + images, not committed to git)
│   ├── dress_train.csv
│   ├── dress_val.csv
│   ├── dress_test.csv
│   └── dress_images/
│       ├── train/
│       ├── val/
│       └── test/
├── train.py                    # Train the Cross-Attention Reranker
├── train_mlp.py                # Train the Global MLP ablation baseline
├── evaluate.py                 # Benchmark evaluation on the test set
├── Architecture.png            # Pipeline diagram
├── requirements.txt
└── README.md
```

---

## How to Run Evaluation & Training

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_dataset.py
```

### 3. (Optional) Precompute Patch Embeddings

Speeds up Cross-Attention Reranker training significantly:

```bash
python scripts/precompute_embeddings.py --data_dir data
```

### 4. Run Full Evaluation via CLI

To benchmark the system on the full test set using the trained reranker:

```bash
# Evaluate with default settings (M=50, alpha=0.5)
python evaluate.py

# Evaluate with custom candidate pool size and blending weight
python evaluate.py --pool_size 50 --alpha 0.5 --checkpoint checkpoints/best_cross_attention_reranker.pt
```

### 5. Run Training via CLI

To train the Cross-Attention Reranker with InfoNCE loss:

```bash
# Fast training on 2,000 pairs for 3 epochs
python train.py --epochs 3 --batch_size 16 --n_train_samples 2000

# Full training on all 12,000 pairs and evaluate on test set
python train.py --epochs 5 --batch_size 16 --eval_after_training
```

### 6. Interactive Notebooks

- **[`notebooks/experiment.ipynb`](notebooks/experiment.ipynb)**:
  - Baseline CLIP zero-shot evaluation ($n=1,197$).
  - Canonical query visualizer (`red floral dress with long sleeves`, `white lace short dress`, `black sleeveless midi dress`).
- **[`notebooks/cross_attention_reranker_experiment.ipynb`](notebooks/cross_attention_reranker_experiment.ipynb)**:
  - Complete end-to-end experiment: token extraction, InfoNCE training, side-by-side visual comparisons, full test set benchmarking, and candidate pool size ($M$) ablation curve.

## Key Takeaways

1. **CLIP Sees the Big Picture, Cross-Attention Checks Details:**  
   Standard CLIP is great at broad ideas like _"red floral dress"_, but compresses the whole picture into one number and misses small details. Cross-attention lets individual words look directly at specific parts of the image (like checking the sleeve length or collar style).

2. **Reranker only ranks the CLIP's top 50 candidates**  
   The reranker only inspects the top 50 candidates found by CLIP. If the right dress is stuck at rank #60, the reranker never even sees it. Stage 1 finds the finalists; Stage 2 picks the winner.This is ok since CLIP's base performance is decent

3. **Combined scoring mechanism**
   It would be bad if we only trusted the Reranker's score since it could easily underfit the small training set, so we are using CLIP's raw score and combing it with the reranker, since CLIP's base performance is good itself, this will provide balance---

## References

1. Radford et al., _Learning Transferable Visual Models From Natural Language Supervision_ (CLIP), ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
2. Han et al., _Fashion200k: An Automatic Approach for Large-Scale Fashion Image Search_, ICCV 2017.
3. Wu et al., _Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback_, CVPR 2021. [arXiv:1905.12794](https://arxiv.org/abs/1905.12794)
4. _Candidate Set Re-ranking for Composed Image Retrieval with Dual Multi-modal Encoder_. [Papers With Code](https://paperswithcode.com/paper/candidate-set-re-ranking-for-composed-image)
