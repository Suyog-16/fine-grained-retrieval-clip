import numpy as np

def compute_retrieval_metrics(ranks, recalls_k=(1, 5, 10)):
    """
    Computes Recall@K, Median Rank, and Mean Rank from a list of 1-indexed ranks.
    """
    n = len(ranks)
    if n == 0:
        return {"recalls": {k: 0.0 for k in recalls_k}, "median_rank": 0, "mean_rank": 0.0, "n": 0}

    recalls = {}
    for k in recalls_k:
        count = sum(1 for r in ranks if r <= k)
        recalls[k] = (count / n) * 100

    return {
        "recalls": recalls,
        "median_rank": int(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "n": n
    }

def format_benchmark_table(base_metrics, rerank_metrics=None, split_name="Test Set"):
    """
    Formats a side-by-side comparison table between Baseline CLIP and Cross-Attention Reranker.
    """
    n = base_metrics["n"]
    lines = []
    lines.append("=" * 65)
    lines.append(f"       RETRIEVAL BENCHMARK EVALUATION ({split_name}, n={n:,})")
    lines.append("=" * 65)

    if rerank_metrics is not None:
        lines.append(f"{'Metric':<14} | {'Baseline CLIP':<15} | {'Cross-Attn Reranker':<20} | {'Delta':<10}")
        lines.append("-" * 65)
        for k in sorted(base_metrics["recalls"].keys()):
            b_pct = base_metrics["recalls"][k]
            r_pct = rerank_metrics["recalls"].get(k, 0.0)
            delta = r_pct - b_pct
            sign = "+" if delta >= 0 else ""
            lines.append(f"Recall@{k:<7} | {b_pct:>13.2f}% | {r_pct:>18.2f}% | {sign}{delta:>7.2f}%")

        lines.append("-" * 65)
        b_med = base_metrics["median_rank"]
        r_med = rerank_metrics["median_rank"]
        lines.append(f"Median Rank    | {b_med:>14} | {r_med:>19} | {r_med - b_med:>+8}")

        b_mean = base_metrics["mean_rank"]
        r_mean = rerank_metrics["mean_rank"]
        lines.append(f"Mean Rank      | {b_mean:>14.1f} | {r_mean:>19.1f} | {r_mean - b_mean:>+8.1f}")
    else:
        lines.append(f"{'Metric':<14} | {'Baseline CLIP':<15}")
        lines.append("-" * 35)
        for k in sorted(base_metrics["recalls"].keys()):
            lines.append(f"Recall@{k:<7} | {base_metrics['recalls'][k]:>13.2f}%")
        lines.append("-" * 35)
        lines.append(f"Median Rank    | {base_metrics['median_rank']:>14}")
        lines.append(f"Mean Rank      | {base_metrics['mean_rank']:>14.1f}")

    lines.append("=" * 65)
    return "\n".join(lines)

def print_benchmark_table(base_metrics, rerank_metrics=None, split_name="Test Set"):
    table_str = format_benchmark_table(base_metrics, rerank_metrics, split_name=split_name)
    print("\n" + table_str + "\n")
