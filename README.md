# Fine-Grained Image Retrieval Using CLIP

This project explores fine-grained image retrieval for dresses using CLIP. The main question is whether deeper cross-modal interaction or attention can improve retrieval precision for fine-grained matching, beyond coarse semantic similarity.

The dataset used here is only "dresses" categories, built from Fashion200k. It includes a data preparation script and notebooks for exploration and experiments.

## Setup

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data

To recreate the dresses dataset from HuggingFace Fashion200k, run `dataset_retieval.py`. It filters the dataset to dresses, keeps one canonical view per product, and saves train/validation/test splits plus images.

## Project Structure

- `dataset_retieval.py` - builds the dataset splits and image folders
- `notebooks/` - data exploration and experiments
- `models/` - model code and baselines
- `data/` - generated CSVs and dress images

## Notes

This repo is notebook-driven, so the experiments are documented in the notebooks rather than a separate training CLI.
