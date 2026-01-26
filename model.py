import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import config

def _mlp(dims, act=nn.ReLU(), dropout=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if act: layers.append(act)
        if dropout > 0: layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def rot6d_to_mat(r6d):
    a1 = r6d[..., 0:3]
    a2 = r6d[..., 3:6]
    
    eps = 1e-8
    
    b1 = F.normalize(a1, dim=-1, eps=eps)
    
    b2_proj = torch.sum(a2 * b1, dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - b2_proj, dim=-1, eps=eps)
    
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack([b1, b2, b3], dim=-2)

class AttnBlock(nn.Module):
    def __init__(self, d_h, d_e, heads, dropout):
        super().__init__()
        self.heads = heads
        self.d_head = d_h // heads
        
        self.to_qkv = nn.Linear(d_h, d_h * 3)
        self.out_proj = nn.Linear(d_h, d_h)
        
        self.edge_bias = nn.Sequential(
            nn.Linear(d_e, heads),
            nn.LayerNorm(heads)
        )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_h * 2 + d_e, d_e),
            nn.ReLU(),
            nn.LayerNorm(d_e),
            nn.Linear(d_e, d_e)
        )
        
        self.norm_h = nn.LayerNorm(d_h)
        self.norm_e = nn.LayerNorm(d_e)
        self.drop_h = nn.Dropout(dropout)
        self.drop_e = nn.Dropout(dropout)

    def forward(self, h, e, mask, knn_mask=None):
        B, N, _ = h.shape
        
        pad_mask = (1.0 - mask.unsqueeze(2).float()) * -1e9
        pad_mask_j = pad_mask.transpose(1, 2)
        pad_mask_full = (pad_mask + pad_mask_j).unsqueeze(1)

        if knn_mask is not None:
            knn_mask_float = knn_mask.unsqueeze(1).float()
            knn_attn_mask = (1.0 - knn_mask_float) * -1e9
            attn_mask = pad_mask_full + knn_attn_mask
        else:
            attn_mask = pad_mask_full

        h_res = h
        h = self.norm_h(h)
        q, k, v = self.to_qkv(h).chunk(3, dim=-1) 
        q = q.view(B, N, self.heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.d_head).transpose(1, 2)
        
        edge_bias = self.edge_bias(self.norm_e(e)).permute(0, 3, 1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        biased_scores = attn_scores + edge_bias
        
        masked_scores = biased_scores + attn_mask
        attn_probs = F.softmax(masked_scores, dim=-1)
        
        h_out = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, N, -1)
        h_out = self.out_proj(h_out)
        h = h_res + self.drop_h(h_out)
        
        e_res = e
        e_normed = self.norm_e(e)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        e_in = torch.cat([h_i, h_j, e_normed], dim=-1)
        e_out = self.edge_mlp(e_in)
        e = e_res + self.drop_e(e_out)
        
        return h, e

class VAE(nn.Module):
    def __init__(self,
                 d_node_in=config.NODE_DIM,
                 d_edge_in=config.EDGE_DIM,
                 d_h=config.GNN_H,
                 d_z=config.GNN_Z,
                 n_layers=config.GNN_LAYERS,
                 n_heads=config.GNN_HEADS,
                 dropout=config.GNN_DROP
                 ):
        super().__init__()
        
        self.feature_adaptor = nn.Sequential(
            nn.Linear(d_node_in, d_h),
            nn.LayerNorm(d_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_h) 
        )
        
        self.edge_in = nn.Linear(d_edge_in, d_h)
        self.enc_layers = nn.ModuleList(
            [AttnBlock(d_h, d_h, n_heads, dropout) for _ in range(n_layers)]
        )
        self.to_mu = nn.Linear(d_h, d_z)
        self.to_logvar = nn.Linear(d_h, d_z)

        self.dec_z_in = nn.Linear(d_z, d_h)
        self.dec_z_edge = nn.Linear(2 * d_z, d_h)
        self.dec_layers = nn.ModuleList(
            [AttnBlock(d_h, d_h, n_heads, dropout) for _ in range(n_layers)]
        )

        self.out_node = nn.Linear(d_h, d_h) 
        self.out_trans = nn.Linear(d_h, 3)
        self.out_rot = nn.Linear(d_h, 6)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, h_adapted, e, mask, knn_mask):
        h_emb = h_adapted 
        e_emb = self.edge_in(e)
        for layer in self.enc_layers:
            h_emb, e_emb = layer(h_emb, e_emb, mask, knn_mask)
        mu = self.to_mu(h_emb)
        logvar = self.to_logvar(h_emb)
        logvar = torch.clamp(logvar, min=config.LOGVAR_MIN, max=config.LOGVAR_MAX)
        z = self.reparam(mu, logvar)
        return mu, logvar, z

    def decode(self, z, N_max, mask):
        B = z.shape[0]
        
        h_dec = self.dec_z_in(z)
        z_i = z.unsqueeze(2).expand(-1, -1, N_max, -1)
        z_j = z.unsqueeze(1).expand(-1, N_max, -1, -1)
        z_ij = torch.cat([z_i, z_j], dim=-1)
        e_dec = self.dec_z_edge(z_ij)
        
        for layer in self.dec_layers:
            h_dec, e_dec = layer(h_dec, e_dec, mask)
            
        h_pred = self.out_node(h_dec)
        t_pred = self.out_trans(h_dec)
        r_pred = self.out_rot(h_dec)
        r_pred_mat = rot6d_to_mat(r_pred)
        
        return h_pred, t_pred, r_pred_mat

    def forward(self, h, e, mask, knn_mask):
        B, N, C_in = h.shape
        h_target = self.feature_adaptor(h)
    
        mu, logvar, z = self.encode(h_target, e, mask, knn_mask)
        
        h_pred, t_pred, r_pred = self.decode(z, N, mask)
        
        return h_pred, t_pred, r_pred, mu, logvar, z, h_target