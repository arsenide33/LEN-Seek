import torch
import torch.nn.functional as F
import config

def _rbf(D, D_min=0.0, D_max=20.0, D_count=32, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view(1, 1, 1, -1)
    D_sigma = (D_max - D_min) / D_count
    K = torch.exp(-((D - D_mu) / D_sigma) ** 2)
    return K

def mat_to_quat(R, eps=1e-8):
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    t = R00 + R11 + R22
    
    q_w = torch.zeros_like(t)
    q_x = torch.zeros_like(t)
    q_y = torch.zeros_like(t)
    q_z = torch.zeros_like(t)
    
    mask_t_pos = t > 0
    if mask_t_pos.any():
        S_pos = torch.sqrt(t[mask_t_pos] + 1.0) * 2
        q_w[mask_t_pos] = 0.25 * S_pos
        q_x[mask_t_pos] = (R21[mask_t_pos] - R12[mask_t_pos]) / S_pos
        q_y[mask_t_pos] = (R02[mask_t_pos] - R20[mask_t_pos]) / S_pos
        q_z[mask_t_pos] = (R10[mask_t_pos] - R01[mask_t_pos]) / S_pos
    
    mask_t_neg_c1 = (t <= 0) & (R00 > R11) & (R00 > R22)
    if mask_t_neg_c1.any():
        S_neg1 = torch.sqrt(1.0 + R00[mask_t_neg_c1] - R11[mask_t_neg_c1] - R22[mask_t_neg_c1]) * 2
        q_w[mask_t_neg_c1] = (R21[mask_t_neg_c1] - R12[mask_t_neg_c1]) / S_neg1
        q_x[mask_t_neg_c1] = 0.25 * S_neg1
        q_y[mask_t_neg_c1] = (R01[mask_t_neg_c1] + R10[mask_t_neg_c1]) / S_neg1
        q_z[mask_t_neg_c1] = (R02[mask_t_neg_c1] + R20[mask_t_neg_c1]) / S_neg1
    
    mask_t_neg_c2 = (t <= 0) & (R11 > R00) & (R11 > R22)
    if mask_t_neg_c2.any():
        S_neg2 = torch.sqrt(1.0 + R11[mask_t_neg_c2] - R00[mask_t_neg_c2] - R22[mask_t_neg_c2]) * 2
        q_w[mask_t_neg_c2] = (R02[mask_t_neg_c2] - R20[mask_t_neg_c2]) / S_neg2
        q_x[mask_t_neg_c2] = (R01[mask_t_neg_c2] + R10[mask_t_neg_c2]) / S_neg2
        q_y[mask_t_neg_c2] = 0.25 * S_neg2
        q_z[mask_t_neg_c2] = (R12[mask_t_neg_c2] + R21[mask_t_neg_c2]) / S_neg2
    
    mask_t_neg_c3 = (t <= 0) & (R22 > R00) & (R22 > R11)
    if mask_t_neg_c3.any():
        S_neg3 = torch.sqrt(1.0 + R22[mask_t_neg_c3] - R00[mask_t_neg_c3] - R11[mask_t_neg_c3]) * 2
        q_w[mask_t_neg_c3] = (R10[mask_t_neg_c3] - R01[mask_t_neg_c3]) / S_neg3
        q_x[mask_t_neg_c3] = (R02[mask_t_neg_c3] + R20[mask_t_neg_c3]) / S_neg3
        q_y[mask_t_neg_c3] = (R12[mask_t_neg_c3] + R21[mask_t_neg_c3]) / S_neg3
        q_z[mask_t_neg_c3] = 0.25 * S_neg3

    q = torch.stack([q_w, q_x, q_y, q_z], dim=-1)
    return F.normalize(q, dim=-1, eps=eps)

def get_edges(T, R, mask_2d, eps=1e-8):
    K = config.K_NEIGHBORS
    B, N, _ = T.shape
    device = T.device

    D_ij = torch.cdist(T, T, p=2)
    
    D_ij_masked = D_ij.clone()
    D_ij_masked.masked_fill_(~mask_2d, float('inf'))
    
    diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    D_ij_masked[diag_mask] = float('inf')
    
    k_safe = min(K, N - 1)
    if k_safe <= 0:
        knn_mask = torch.zeros_like(mask_2d)
    else:
        _, top_k_indices = torch.topk(D_ij_masked, k=k_safe, dim=-1, largest=False)
        knn_mask = torch.zeros_like(mask_2d)
        knn_mask.scatter_(dim=-1, index=top_k_indices, value=True)
    
    knn_mask = knn_mask | knn_mask.transpose(-1, -2)
    
    final_edge_mask = (mask_2d & knn_mask)
    final_edge_mask_3d = final_edge_mask.unsqueeze(-1)

    rbf_feats = _rbf(D_ij.unsqueeze(-1), device=device)
    
    T_i = T.unsqueeze(2).expand(-1, -1, N, -1)
    T_j = T.unsqueeze(1).expand(-1, N, -1, -1)
    D_vec = T_j - T_i
    dir_feats = F.normalize(D_vec, dim=-1, eps=eps)
    
    R_i_T = R.transpose(-1, -2).unsqueeze(2).expand(-1, -1, N, -1, -1)
    R_j = R.unsqueeze(1).expand(-1, N, -1, -1, -1)
    R_rel = torch.matmul(R_i_T, R_j)
    quat_feats = mat_to_quat(R_rel)

    e_batch = torch.cat([rbf_feats, dir_feats, quat_feats], dim=-1)
    
    e_masked = e_batch * final_edge_mask_3d
    
    return e_masked, final_edge_mask