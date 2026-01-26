import os
import sys
import glob
import numpy as np
import pickle
import gc
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs

import config
from model import VAE
from loss import vae_loss
from utils.losstrack import LossTracker
from utils.geometry import get_edges

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def aug_norm(h, sc_vecs, trans, mask, rots, res_ids, chain_ids, eps=1e-6):
    B, N, _ = trans.shape
    mask_float = mask.float().unsqueeze(-1)
    num_valid = mask_float.sum(dim=1, keepdim=True).clamp(min=eps)
    T_mean = (trans * mask_float).sum(dim=1, keepdim=True) / num_valid
    t_norm = (trans - T_mean) * mask_float
    r_norm = rots

    return h, sc_vecs, t_norm, mask, r_norm, res_ids, chain_ids


def load_data(accelerator, dataset_path):
    if not os.path.exists(dataset_path):
        if accelerator.is_main_process:
            print(f"[train.py] Error: Dataset not found: {dataset_path}", flush=True)
        sys.exit(1)

    if accelerator.is_main_process:
        accelerator.print(f"[{accelerator.process_index}] Main process loading dataset (to cache): {dataset_path}", flush=True)
        try:
            file_size = os.path.getsize(dataset_path)
            with open(dataset_path, 'rb') as f:
                with tqdm.wrapattr(f, "read", total=file_size, desc="[Rank 0] Caching", leave=False) as pbar:
                    pickle.load(pbar)
            accelerator.print(f"\n[{accelerator.process_index}] Main process caching complete.", flush=True)
        except Exception as e:
            accelerator.print(f"[{accelerator.process_index}] Error during main load: {e}", flush=True)
            sys.exit(1)
            
    accelerator.wait_for_everyone()

    accelerator.print(f"[{accelerator.process_index}] Loading dataset (from cache)...", flush=True)
    try:
        file_size = os.path.getsize(dataset_path)
        with open(dataset_path, 'rb') as f:
            if accelerator.is_main_process:
                with tqdm.wrapattr(f, "read", total=file_size, desc="[Rank 0] Final Load", leave=False) as pbar:
                    dataset = pickle.load(pbar)
                print("") 
            else:
                dataset = pickle.load(f)
                
        accelerator.print(f"[{accelerator.process_index}] Loaded {len(dataset)} items.", flush=True)
    except Exception as e:
        accelerator.print(f"[{accelerator.process_index}] Error: {e}", flush=True)
        sys.exit(1)

    return dataset


def get_beta(epoch):
    if epoch < config.KL_START:
        return 0.0
    total_anneal_epochs = config.EPOCHS - config.KL_START
    cycle_len = total_anneal_epochs // config.KL_CYCLES
    if cycle_len == 0:
        return config.BETA_MAX
    cycle_epoch = (epoch - config.KL_START) % cycle_len
    beta = min(1.0, cycle_epoch / (cycle_len * 0.5)) * config.BETA_MAX
    if (epoch - config.KL_START) // cycle_len >= config.KL_CYCLES:
        beta = config.BETA_MAX
    return beta


def get_lr(epoch):
    frac = epoch / config.EPOCHS
    return config.START_LR * (1 - frac) + config.END_LR * frac


def save_model(model, optimizer, epoch, file_path, accelerator):
    if accelerator.is_main_process:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        model_state_dict = accelerator.unwrap_model(model).state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }, file_path)
        
        accelerator.print(f"[train.py] Checkpoint saved to {file_path}", flush=True)


def train_epoch(model, dataloader, optimizer, accelerator, kl_beta):
    model.train()
    tracker = LossTracker()
    
    if accelerator.is_main_process:
        accelerator.print(f"[Rank 0] Starting training epoch, batches: {len(dataloader)}", flush=True)
        pbar = tqdm(dataloader, desc="Training", leave=False)
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        h_true, sc_vecs, trans, mask, rots, res_ids, chain_ids = batch
        
        h_true, sc_vecs, t, mask, r, res_ids, chain_ids = aug_norm(h_true, sc_vecs, trans, mask, rots, res_ids, chain_ids)
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
        e, knn_mask = get_edges(t, r, mask_2d)

        optimizer.zero_grad()
        
        h_pred, t_pred, r_pred, mu, logvar, z, h_target = model(h_true, e, mask, knn_mask)
        
        losses = vae_loss(h_true=h_target, t_true=t, r_true=r,
                          h_pred=h_pred, t_pred=t_pred, r_pred=r_pred,
                          mu=mu, logvar=logvar, mask=mask, kl_beta=kl_beta)

        total_loss = losses['total_loss']
        if torch.isnan(total_loss):
            if accelerator.is_main_process:
                print(f"[train.py] Error: NaN detected in loss. Skipping batch.", flush=True)
            continue

        accelerator.backward(total_loss)
        
        if config.MAX_GRAD > 0:
            accelerator.clip_grad_norm_(model.parameters(), config.MAX_GRAD)
        
        optimizer.step()

        loss_items = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        tracker.update(loss_items) 

        if accelerator.is_main_process:
            pbar.set_postfix({
                'Loss': f"{loss_items['total_loss']:.3f}",
                'KL': f"{loss_items['kl_loss']:.3f}",
                'FAPE': f"{loss_items['recon_fape']:.3f}",
                'RMSD': f"{loss_items['recon_rmsd']:.3f}"
            })

    return tracker.get_avg_losses()


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    data_path = getattr(config, 'TRAIN_DATA_PATH', getattr(config, 'DATA_PATH', None))
    dataset = load_data(accelerator, data_path)
    if len(dataset) == 0:
        return

    accelerator.print(f"[{accelerator.process_index}] Model init", flush=True)
    model = VAE()
    
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, 
                            num_workers=0, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.START_LR)
    
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    save_file_path = config.CKPT_PATH 

    for epoch in range(1, config.EPOCHS + 1):
        kl_beta = get_beta(epoch)
        current_lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        avg_losses = train_epoch(model, dataloader, optimizer, accelerator, kl_beta) 

        if accelerator.is_main_process:
            
            print(f"\nEpoch {epoch}: LR={current_lr:.6f}, Beta={kl_beta:.4f}", flush=True)
            for k, v in avg_losses.items():
                print(f"  {k}: {v:.4f}", flush=True)
            
            if epoch == config.EPOCHS:
                save_model(model, optimizer, epoch, save_file_path, accelerator)

if __name__ == "__main__":
    main()