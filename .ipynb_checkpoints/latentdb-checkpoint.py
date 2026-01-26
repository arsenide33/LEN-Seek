import os
import sys
import numpy as np
import torch
import pickle
from tqdm import tqdm
import gc
import torch.distributed as dist
from accelerate import Accelerator
import argparse
import traceback

try:
    import config
    from model import VAE
    from utils.geometry import get_edges
    from utils.bsdb import PDBDataParallel 
except ImportError as e:
    print(f"[latentdb.py] Error: Import failed: {e}")
    sys.exit(1)

def load_data(accelerator, data_path):
    if not os.path.exists(data_path):
        if accelerator.is_main_process:
            print(f"[latentdb.py] Error: Dataset not found: {data_path}")
        sys.exit(1)

    dataset = None
    
    if accelerator.is_main_process:
        print(f"[latentdb.py] Loading dataset: {data_path}")
        try:
            with open(data_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"[latentdb.py] Loaded {len(dataset.data)} items.")
        except Exception as e:
            print(f"[latentdb.py] Error: loading pickle: {e}")
            dataset = None
            
    if accelerator.num_processes > 1:
        obj_list = [dataset]
        dist.broadcast_object_list(obj_list, src=0)
        dataset = obj_list[0]
    
    if dataset is None:
        sys.exit(1)

    return dataset

def load_model(model_path, device):
    print(f"[latentdb.py] Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"[latentdb.py] Error: Model file not found at {model_path}")
        sys.exit(1)
        
    model = VAE().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(cleaned_state)
    model.eval()
    return model

def prep_batch(batch_items, device):
    N_max = config.NUM_PTS
    
    h_list, T_list, R_list, mask_list = [], [], [], []
    valid_indices = []
    
    error_counter = 0
    
    for idx, ftr_data in enumerate(batch_items):
        try:
            node_feats, rots, sc_vecs, trans, res_nums, chain_ids = ftr_data
            
            n_res = len(node_feats)
            
            if n_res >= N_max:
                poc_idx = np.arange(N_max)
                mask_np = np.ones(N_max, dtype=bool)
            else:
                poc_idx = np.arange(n_res)
                pad = N_max - n_res
                poc_idx = np.pad(poc_idx, (0, pad), mode='wrap')
                mask_np = np.ones(N_max, dtype=bool)
                mask_np[-pad:] = False
            
            h_list.append(torch.from_numpy(node_feats[poc_idx]).float())
            T_list.append(torch.from_numpy(trans[poc_idx]).float())
            R_list.append(torch.from_numpy(rots[poc_idx]).float())
            mask_list.append(torch.from_numpy(mask_np))
            valid_indices.append(idx)
            
        except Exception as e:
            error_counter += 1
            if error_counter == 1: 
                print(f"\n[latentdb.py] DEBUG: prep_batch error sample: {e}")
            continue
    
    if not valid_indices:
        if error_counter > 0:
            print(f"[latentdb.py] Error: All {len(batch_items)} items in batch failed to prep.")
        return None, None, None, None, []
    
    h = torch.stack(h_list).to(device)
    T = torch.stack(T_list).to(device)
    R = torch.stack(R_list).to(device)
    mask = torch.stack(mask_list).to(device)
    
    mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
    e, knn_mask = get_edges(T, R, mask_2d)
    
    return h, e, mask, knn_mask, valid_indices

@torch.no_grad()
def encode_batch(model, h, e, mask, knn_mask):
    h_adapted = model.feature_adaptor(h)
    mu, _, _ = model.encode(h_adapted, e, mask, knn_mask)
    return mu

def main():
    parser = argparse.ArgumentParser(description="Generate Latent Database using VAE")
    parser.add_argument('--bsize', type=int, default=getattr(config, 'BATCH_SIZE', 64), 
                        help=f"Batch size (Default: {getattr(config, 'BATCH_SIZE', 64)})")
    args = parser.parse_args()

    model_path = config.CKPT_PATH
    input_data_path = config.TRAIN_DATA_PATH 
    output_db_path = config.LATENT_DATA_PATH

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    
    if accelerator.is_main_process:
        print(f"[latentdb.py]")
        print(f"Device: {device}, Processes: {accelerator.num_processes}")
        print(f"Model Path : {model_path}")
        print(f"Input Data : {input_data_path}")
        print(f"Output DB  : {output_db_path}")
        print(f"Batch Size : {args.bsize}")

    model = load_model(model_path, device)
    dataset = load_data(accelerator, input_data_path)

    all_data = dataset.data
    all_paths = dataset.path
    
    with accelerator.split_between_processes(all_data) as split_data:
        rank_data = list(split_data)
    with accelerator.split_between_processes(all_paths) as split_paths:
        rank_paths = list(split_paths)
        
    if accelerator.is_main_process:
        print(f"[latentdb.py] Rank {rank} processing {len(rank_data)} items.")

    db_list = []
    success_total = 0
    fail_total = 0
    
    iterable = range(0, len(rank_data), args.bsize)
    if accelerator.is_main_process:
        pbar = tqdm(iterable, desc="Encoding DB")
    else:
        pbar = iterable
    
    for start_idx in pbar:
        end_idx = min(start_idx + args.bsize, len(rank_data))
        batch_items = rank_data[start_idx:end_idx]
        batch_paths = rank_paths[start_idx:end_idx]
        
        h, e, mask, knn_mask, v_idx = prep_batch(batch_items, device)
        
        if h is None: 
            fail_total += len(batch_items)
            continue
        
        try:
            mu = encode_batch(model, h, e, mask, knn_mask)
            mu_np = mu.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for i, valid_i in enumerate(v_idx):
                db_list.append((
                    mu_np[i:i+1], 
                    mask_np[i:i+1], 
                    os.path.abspath(batch_paths[valid_i])
                ))
            
            success_total += len(v_idx)
            fail_total += (len(batch_items) - len(v_idx))
            
        except RuntimeError as e:
            print(f"\n[latentdb.py] Error: Batch encoding failed: {e}")
            fail_total += len(batch_items)
            gc.collect()
            torch.cuda.empty_cache()
            continue

    if accelerator.num_processes > 1:
        gathered_db = [None] * accelerator.num_processes
        dist.gather_object(db_list, gathered_db if rank == 0 else None, dst=0)
        if rank == 0:
            final_db = []
            for sublist in gathered_db:
                final_db.extend(sublist)
    else:
        final_db = db_list

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"[latentdb.py]")
        print(f"Latent DB Generation Complete")
        print(f"Total entries in DB: {len(final_db)}")
        print(f"Success items (this rank): {success_total}")
        print(f"Failed items (this rank): {fail_total}")
        if len(final_db) > 0:
            os.makedirs(os.path.dirname(output_db_path), exist_ok=True)
            with open(output_db_path, 'wb') as f:
                pickle.dump(final_db, f)
            print(f"[latentdb.py] Saved Latent DB to: {output_db_path}")
        else:
            print("[latentdb.py] Error: No entries were generated. Check debug logs above.")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()