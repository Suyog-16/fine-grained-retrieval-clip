import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMLPReranker(nn.Module):
    """Ablation baseline: MLP operating only on concatenated global CLIP embeddings.

    Input:
        global_text_embed : [B, 512]  - CLIP encode_text output
        global_img_embed  : [B, 512]  - CLIP encode_image output
    Output:
        score             : [B]       - scalar matching score per pair
    """
    def __init__(self, embed_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_text_embed, global_img_embed):
        t_norm = F.normalize(global_text_embed.float(), dim=-1)
        v_norm = F.normalize(global_img_embed.float(), dim=-1)
        feat = torch.cat([t_norm, v_norm, t_norm * v_norm], dim=-1)
        score = self.mlp(feat).squeeze(-1)
        return score
