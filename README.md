# Fine-Grained Dress Retrieval — CLIP + Cross-Attention Reranking

Text-to-image retrieval on fashion dresses using **OpenAI CLIP (`ViT-B/32`)** with a custom **Cross-Attention Reranker**. CLIP alone struggles with fine-grained attributes (sleeve length, lace texture, neckline cuts) since it compresses an image into a single 512-d vector. This project adds a second stage that uses token-level attention to pick up those details.

---

## System Architecture

### Stage 1 — Coarse Retrieval (CLIP)

![Stage 1 Coarse Retrieval](coarse.drawio.svg)

The full image gallery is pre-encoded into 512-d CLIP vectors. For any text query, dot-product similarity retrieves the **Top-M candidates** (M = 50) in milliseconds.

### Stage 2 — Cross-Attention Reranking

![Stage 2 Cross-Attention Reranking](cross_attention.drawio.svg)

For each Top-M candidate, text tokens `[77 × 512]` attend over image patch tokens `[50 × 512]` via Multi-Head Cross-Attention. The output is mean-pooled and passed through an MLP to produce a fine-grained relevance score. Final score blends both:

$$\text{Final Score} = \alpha \cdot s_{\text{rerank}} + (1 - \alpha) \cdot s_{\text{clip}} \quad (\alpha = 0.5)$$

---

## Results

Evaluated on **1,197 dress queries** against a 1,197-image test gallery. Metrics: Recall@K (higher is better), Median/Mean Rank (lower is better).

| Metric          | Baseline CLIP | Cross-Attention Reranker | Δ |
| :-------------- | :-----------: | :----------------------: | :-: |
| **Recall@1**    | 11.70%        | **12.53%**               | +0.84% |
| **Recall@5**    | 29.82%        | **32.08%**               | +2.26% |
| **Recall@10**   | 42.11%        | **45.36%**               | +3.26% |
| **Median Rank** | 15            | **13**                   | -2 |
| **Mean Rank**   | 49.5          | **48.1**                 | -1.4 |

> The reranker shows the most gain at R@5 and R@10, pulling relevant dresses up from ranks 10–25 into the top results.

---

## Dataset

Built from **[Fashion200k](https://github.com/xthan/fashion-200k)**, filtered to dresses only, deduplicated to one view per product:

| Split | Size | File |
| :---- | :--- | :--- |
| Train | 12,000 | `data/dress_train.csv` |
| Val   | 1,500  | `data/dress_val.csv` |
| Test  | 1,197  | `data/dress_test.csv` |

Download from Hugging Face:
```bash
python scripts/download_dataset.py
```

---

## Setup & Usage

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Evaluate** (runs baseline CLIP + reranker benchmark):
```bash
python evaluate.py
python evaluate.py --pool_size 50 --alpha 0.5  # custom settings
```

**Train** the Cross-Attention Reranker (InfoNCE loss):
```bash
python train.py --epochs 3 --batch_size 16 --n_train_samples 2000  # quick run
python train.py --epochs 5 --batch_size 16 --eval_after_training    # full run
```

**Precompute** patch embeddings (speeds up training):
```bash
python scripts/precompute_embeddings.py --data_dir data
```

**Notebooks:**
- [`notebooks/experiment.ipynb`](notebooks/experiment.ipynb) — Baseline CLIP evaluation + query visualizer
- [`notebooks/cross_attention_reranker_experiment.ipynb`](notebooks/cross_attention_reranker_experiment.ipynb) — Full end-to-end experiment with ablations

---

## Key Takeaways

1. **CLIP misses fine-grained details** — it's great at broad concepts but collapses the whole image into one vector, losing spatial info like sleeve length or collar style.
2. **Reranker only sees Top-50** — if the correct dress isn't in CLIP's Top-50, the reranker never sees it. Stage 1 quality sets the ceiling.
3. **Score blending matters** — using only the reranker score risks underfitting on a small training set. Blending with CLIP's score keeps results stable.

---

## References

1. Radford et al., _CLIP_, ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
2. Han et al., _Fashion200k_, ICCV 2017.
3. Wu et al., _Fashion IQ_, CVPR 2021. [arXiv:1905.12794](https://arxiv.org/abs/1905.12794)
4. _Candidate Set Re-ranking for Composed Image Retrieval_. [Papers With Code](https://paperswithcode.com/paper/candidate-set-re-ranking-for-composed-image)
