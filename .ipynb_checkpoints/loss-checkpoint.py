import torch
import torch.nn.functional as F
import config

def _check_nan(tensor, name):
    if torch.isnan(tensor).any():
        error_msg = f"\n!!! [loss.py] NaN DETECTED in loss: [ {name} ] !!!\n"
        print(error_msg)
        raise RuntimeError(error_msg)
    return False

def _masked_mean(tensor, mask, dim=None, eps=1e-6):
    mask_float = mask.float()
    if tensor.dim() > mask_float.dim() and tensor.shape[:2] == mask_float.shape:
        mask_float = mask_float.unsqueeze(-1)
    num_valid = mask_float.sum(dim=dim, keepdim=True if dim is not None else False).clamp(min=eps)
    masked_tensor_sum = (tensor * mask_float).sum(dim=dim, keepdim=True if dim is not None else False)
    return masked_tensor_sum / num_valid

def ankh_cos(pred, true, mask, eps=1e-8):
    cos_sim = F.cosine_similarity(pred, true, dim=-1, eps=eps)
    loss_per_residue = 1.0 - cos_sim
    return _masked_mean(loss_per_residue, mask)

def ankh_norm(pred, true, mask):
    true_norm = torch.norm(true, p=2, dim=-1)
    recon_norm = torch.norm(pred, p=2, dim=-1)
    norm_diff = torch.abs(true_norm - recon_norm)
    return _masked_mean(norm_diff, mask)

def kabsch_rmsd(coords_pred, coords_true, mask, eps=1e-8):
    mask_float = mask.float().unsqueeze(-1)
    
    num_valid = mask_float.sum(dim=1, keepdim=True).clamp(min=eps)
    
    center_true = (coords_true * mask_float).sum(dim=1, keepdim=True) / num_valid
    center_pred = (coords_pred * mask_float).sum(dim=1, keepdim=True) / num_valid
    
    coords_true_c = (coords_true - center_true) * mask_float
    coords_pred_c = (coords_pred - center_pred) * mask_float
    
    H = torch.matmul(coords_pred_c.transpose(1, 2), coords_true_c)
    
    try:
        U, S, Vt = torch.linalg.svd(H)
    except torch.linalg.LinAlgError as e:
        print(f"[loss.py] Error: SVD failed in Kabsch: {e}. Skipping RMSD batch.")
        return torch.tensor(0.0, device=coords_pred.device, dtype=coords_pred.dtype)

    d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    S_diag = torch.diag_embed(torch.ones_like(S))
    S_diag[:, -1, -1] = d
    
    R = torch.matmul(Vt.transpose(1, 2), torch.matmul(S_diag, U.transpose(1, 2)))
    
    coords_pred_aligned = torch.matmul(coords_pred_c, R)
    
    sq_diff = (coords_pred_aligned - coords_true_c).pow(2).sum(dim=-1)
    
    mse = sq_diff[mask].sum() / mask_float.sum().clamp(min=eps)
    rmsd = torch.sqrt(mse + eps)
    
    return rmsd
    

def fape_loss(pred_T, pred_R, true_T, true_R, mask, eps=1e-8, clamp_dist=10.0):
    B, N, _ = pred_T.shape
    
    T_true_i = true_T.unsqueeze(2).expand(B, N, N, 3)
    T_true_j = true_T.unsqueeze(1).expand(B, N, N, 3)
    
    T_pred_i = pred_T.unsqueeze(2).expand(B, N, N, 3)
    T_pred_j = pred_T.unsqueeze(1).expand(B, N, N, 3)
    
    R_true_i_T = true_R.transpose(-1, -2).unsqueeze(2).expand(B, N, N, 3, 3)
    R_pred_i_T = pred_R.transpose(-1, -2).unsqueeze(2).expand(B, N, N, 3, 3)
    
    T_diff_true_global = (T_true_j - T_true_i).unsqueeze(-1)
    T_diff_pred_global = (T_pred_j - T_pred_i).unsqueeze(-1)
    
    T_true_local = torch.matmul(R_true_i_T, T_diff_true_global).squeeze(-1)
    T_pred_local = torch.matmul(R_pred_i_T, T_diff_pred_global).squeeze(-1)
    
    dist_error = torch.norm(T_true_local - T_pred_local, dim=-1, p=2)
    
    clamped_dist_error = torch.clamp(dist_error, max=clamp_dist)
    
    mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
    
    num_valid_j = mask.float().sum(dim=1, keepdim=True).clamp(min=eps)
    clamped_dist_error_masked = clamped_dist_error * mask_2d.float()
    
    fape_per_i = clamped_dist_error_masked.sum(dim=2) / num_valid_j
    
    fape_loss_val = _masked_mean(fape_per_i, mask)
    
    return fape_loss_val

def vae_loss(
    h_true, t_true, r_true,
    h_pred, t_pred, r_pred,
    mu, logvar, mask, kl_beta
):
    losses = {}

    _check_nan(mu, 'mu')
    _check_nan(logvar, 'logvar')
    _check_nan(h_pred, 'h_pred')
    _check_nan(t_pred, 't_pred')
    _check_nan(r_pred, 'r_pred')

    mu_clamped = torch.clamp(mu, min=-20, max=20)
    logvar_clamped = torch.clamp(logvar, min=-20, max=20)
    
    kl_per_node_per_dim = -0.5 * (
        1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp()
    )
    kl_per_node = kl_per_node_per_dim.sum(dim=-1)
    
    kl_per_node_clamped = torch.clamp(kl_per_node, min=config.FREE_BITS)
    
    losses['kl_loss'] = _masked_mean(kl_per_node_clamped, mask)
    losses['kl_loss_raw'] = _masked_mean(kl_per_node, mask)
    _check_nan(losses['kl_loss'], 'kl_loss')

    ankh_mse_per_residue = F.mse_loss(h_pred, h_true, reduction='none').mean(dim=-1)
    losses['recon_ankh_mse'] = _masked_mean(ankh_mse_per_residue, mask)
    losses['recon_ankh_cos'] = ankh_cos(h_pred, h_true, mask)
    losses['recon_ankh_norm'] = ankh_norm(h_pred, h_true, mask)

    losses['recon_fape'] = fape_loss(
        t_pred, r_pred,
        t_true, r_true,
        mask
    )
    
    losses['recon_rmsd'] = kabsch_rmsd(
        t_pred, t_true, mask
    )
    
    total_loss = 0.0
    total_loss += losses['kl_loss'] * kl_beta * config.W_KL
    
    total_loss += losses['recon_ankh_mse'] * config.W_ANKH_MSE
    total_loss += losses['recon_ankh_cos'] * config.W_ANKH_COS
    total_loss += losses['recon_ankh_norm'] * config.W_ANKH_NORM
    
    total_loss += losses['recon_fape'] * config.W_FAPE
    total_loss += losses['recon_rmsd'] * config.W_RMSD

    losses['total_loss'] = total_loss
    _check_nan(losses['total_loss'], 'total_loss')

    return losses
