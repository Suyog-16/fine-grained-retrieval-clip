import torch
import clip
from tqdm.auto import tqdm
from .models import extract_text_tokens, extract_patch_tokens
from .metrics import compute_retrieval_metrics, print_benchmark_table

class TwoStageRetriever:
    """
    Two-Stage Retrieval System:
    - Stage 1: Coarse candidate search with CLIP global embeddings
    - Stage 2: Fine-grained token-level cross-attention reranking
    """
    def __init__(self, clip_model, reranker=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.clip_model = clip_model.to(device).eval()
        self.reranker = reranker.to(device).eval() if reranker is not None else None
        self.device = device

        self.gallery_embeddings = None  # [N, 512]
        self.gallery_ids = None         # list of str IDs
        self.id_to_idx = None           # dict: str ID -> index
        self.patch_cache = {}           # dict: str ID -> patch tensor [50, 512]

    def build_gallery(self, loader, cache_patches=True, desc="Encoding gallery"):
        """
        Precomputes global CLIP embeddings and optional patch tokens for gallery images.
        """
        all_embeds = []
        all_ids = []
        self.patch_cache = {}

        with torch.no_grad():
            for images, ids in tqdm(loader, desc=desc):
                images = images.to(self.device)

                # 1. Global CLIP embeddings
                feats = self.clip_model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embeds.append(feats.cpu())
                all_ids.extend([str(i) for i in ids])

                # 2. Patch embeddings for reranking
                if cache_patches:
                    patches = extract_patch_tokens(self.clip_model, images).cpu()
                    for item_id, patch in zip(ids, patches):
                        self.patch_cache[str(item_id)] = patch

        self.gallery_embeddings = torch.cat(all_embeds).to(self.device)
        self.gallery_ids = all_ids
        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(all_ids)}
        print(f"Gallery built: {self.gallery_embeddings.shape[0]:,} images indexed.")

    def retrieve_baseline(self, query_text, top_k=6):
        """Baseline coarse CLIP retrieval."""
        if self.gallery_embeddings is None:
            raise ValueError("Gallery not built yet. Call build_gallery() first.")

        with torch.no_grad():
            tokens = clip.tokenize([query_text], truncate=True).to(self.device)
            text_feat = self.clip_model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores = (self.gallery_embeddings @ text_feat.T).squeeze(1)
        top_idx = scores.topk(min(top_k, len(self.gallery_ids))).indices.tolist()
        return [(self.gallery_ids[i], scores[i].item()) for i in top_idx]

    def retrieve_reranked(self, query_text, top_k=6, candidate_pool_size=50, alpha=0.5):
        """Two-stage retrieval with cross-attention reranking."""
        if self.gallery_embeddings is None:
            raise ValueError("Gallery not built yet. Call build_gallery() first.")
        if self.reranker is None:
            return self.retrieve_baseline(query_text, top_k)

        with torch.no_grad():
            tokens = clip.tokenize([query_text], truncate=True).to(self.device)
            text_feat = self.clip_model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            # Stage 1: Coarse retrieval
            base_scores = (self.gallery_embeddings @ text_feat.T).squeeze(1)
            pool_size = min(candidate_pool_size, len(self.gallery_ids))
            top_candidates = base_scores.topk(pool_size).indices.tolist()

            # Stage 2: Cross-attention reranking
            query_toks = extract_text_tokens(self.clip_model, tokens)  # [1, 77, 512]
            query_toks_expanded = query_toks.expand(len(top_candidates), -1, -1)

            cand_patches = torch.stack([
                self.patch_cache[self.gallery_ids[i]] for i in top_candidates
            ]).to(self.device)

            rerank_scores = self.reranker(query_toks_expanded, cand_patches)

            # Min-max normalization for blending
            b_pool = base_scores[top_candidates]
            b_norm = (b_pool - b_pool.min()) / (b_pool.max() - b_pool.min() + 1e-8)
            r_norm = (rerank_scores - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min() + 1e-8)

            blended = alpha * r_norm + (1.0 - alpha) * b_norm
            rerank_sub_order = blended.argsort(descending=True).tolist()

            final_top_idx = [top_candidates[i] for i in rerank_sub_order[:top_k]]
            return [(self.gallery_ids[i], blended[rerank_sub_order[pos]].item()) for pos, i in enumerate(final_top_idx)]

    def evaluate(self, df, candidate_pool_size=50, alpha=0.5, text_col="text", split_name="Test Set"):
        """
        Runs complete benchmark evaluation comparing Baseline CLIP and Reranker.
        """
        base_ranks = []
        rerank_ranks = []

        if self.reranker is not None:
            self.reranker.eval()

        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating ({split_name})"):
                target_id = str(row["item_ID"])
                correct_idx = self.id_to_idx.get(target_id)
                if correct_idx is None:
                    continue

                # Stage 1 text encoding & base ranking
                tokens = clip.tokenize([str(row[text_col])], truncate=True).to(self.device)
                text_feat = self.clip_model.encode_text(tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

                base_scores = (self.gallery_embeddings @ text_feat.T).squeeze(1)
                base_ranked_indices = base_scores.argsort(descending=True).tolist()

                b_rank = base_ranked_indices.index(correct_idx) + 1
                base_ranks.append(b_rank)

                # Stage 2 cross-attention reranking (if reranker is loaded)
                if self.reranker is not None:
                    pool_indices = base_ranked_indices[:candidate_pool_size]
                    query_toks = extract_text_tokens(self.clip_model, tokens).expand(len(pool_indices), -1, -1)
                    cand_patches = torch.stack([
                        self.patch_cache[self.gallery_ids[i]] for i in pool_indices
                    ]).to(self.device)

                    rerank_scores = self.reranker(query_toks, cand_patches)

                    b_pool = base_scores[pool_indices]
                    b_norm = (b_pool - b_pool.min()) / (b_pool.max() - b_pool.min() + 1e-8)
                    r_norm = (rerank_scores - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min() + 1e-8)
                    blended = alpha * r_norm + (1.0 - alpha) * b_norm

                    rerank_sub_order = blended.argsort(descending=True).tolist()
                    reranked_top_m = [pool_indices[i] for i in rerank_sub_order]
                    final_ranked_indices = reranked_top_m + base_ranked_indices[candidate_pool_size:]

                    r_rank = final_ranked_indices.index(correct_idx) + 1
                    rerank_ranks.append(r_rank)

        base_metrics = compute_retrieval_metrics(base_ranks)
        rerank_metrics = compute_retrieval_metrics(rerank_ranks) if self.reranker is not None else None

        print_benchmark_table(base_metrics, rerank_metrics, split_name=split_name)
        return {
            "base_metrics": base_metrics,
            "rerank_metrics": rerank_metrics,
            "base_ranks": base_ranks,
            "rerank_ranks": rerank_ranks
        }
