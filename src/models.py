import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_text_tokens(model, tokens):
    """
    Extracts contextualized text token embeddings from OpenAI CLIP text encoder.
    Args:
        model: Loaded CLIP model
        tokens: Tokenized query tensor [B, 77]
    Returns:
        Tensor of shape [B, 77, 512]
    """
    x = model.token_embedding(tokens).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x).type(model.dtype)
    if model.text_projection is not None:
        x = x @ model.text_projection
    return x

def extract_patch_tokens(model, images):
    """
    Extracts patch token embeddings from the CLIP ViT-B/32 visual encoder.
    Args:
        model: Loaded CLIP model
        images: Preprocessed image tensor [B, 3, 224, 224]
    Returns:
        Tensor of shape [B, 50, 512] (1 CLS + 49 spatial patch tokens)
    """
    v = model.visual
    x = images.type(model.dtype)
    x = v.conv1(x)  # [B, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid**2]
    x = x.permute(0, 2, 1)  # [B, grid**2, width]

    cls_token = v.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)  # [B, grid**2 + 1, width]
    x = x + v.positional_embedding.to(x.dtype)
    x = v.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = v.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = v.ln_post(x)
    if v.proj is not None:
        x = x @ v.proj
    return x

class CrossAttentionReranker(nn.Module):
 
    def __init__(self, embed_dim=512, num_heads=8, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, text_tokens, patch_tokens):
        # Cast to float32 for stable multi-head attention and gradients
        t = text_tokens.float()
        p = patch_tokens.float()

        attn_out, _ = self.cross_attn(query=t, key=p, value=p)
        x = self.norm(t + attn_out)
        pooled = x.mean(dim=1)
        score = self.mlp(pooled).squeeze(-1)
        return score
