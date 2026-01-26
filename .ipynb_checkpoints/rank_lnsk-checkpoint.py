import os
import sys
import numpy as np
import torch
from accelerate import Accelerator 
import pickle
import gc
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd 
import glob

import config
from model import VAE
from loss import vae_loss
from utils.geometry import get_edges
from embed import get_chain_sequences 
import utils.featurize as featurize 
from utils.pdb_parser import parse_pdb

MODEL_PATH = config.CKPT_PATH
DB_PATH = config.LATENT_DATA_PATH
QUERY_DIR = config.EVAL_DB_PATH
ANKH_PATH = config.EVAL_ANKH_PATH

COMP_BATCH = 64 
CSV_DIR = f"./comps/"

def load_model(model_path, device):
    print(f"[rank_lnsk.py]Loading model from: {model_path}")
    model = VAE().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model

def encode_query(model, protein_path, pocket_path, device, ankh_path_dir, verbose=False):
    if not os.path.exists(protein_path):
        if verbose: print(f"[rank_lnsk.py] Error: Protein file not found: {protein_path}")
        return None, None, None
    if not os.path.exists(pocket_path):
        if verbose: print(f"[rank_lnsk.py] Error: Pocket file not found: {pocket_path}")
        return None, None, None

    pdbid = os.path.basename(os.path.dirname(protein_path))
    os.makedirs(ankh_path_dir, exist_ok=True)
    
    chain_seqs = get_chain_sequences(protein_path)
    if not chain_seqs:
        if verbose: print(f"[rank_lnsk.py] Error: No valid chain sequences found in {protein_path}")
        return None, None, None

    tasks = [] 
    for chain_id, sequence in chain_seqs.items():
        out_path = os.path.join(ankh_path_dir, f"{pdbid}_chain_{chain_id}.npy")
        should_generate = False
        if not os.path.exists(out_path):
            should_generate = True
        else:
            try:
                emb = np.load(out_path)
                if emb.shape[0] != len(sequence):
                    if verbose: 
                        print(f"[rank_lnsk.py] Length mismatch for {pdbid} Chain {chain_id}. Regenerating.")
                    should_generate = True
                del emb
            except Exception:
                should_generate = True

        if should_generate:
            tasks.append((chain_id, sequence, out_path))
    
    if tasks:
        if verbose: print(f"[rank_lnsk.py] Generating {len(tasks)} missing/mismatched embeddings for {pdbid}...")
        model_ankh, tokenizer_ankh = featurize.load_ankh(device)
        if model_ankh is None:
            if verbose: print(f"[rank_lnsk.py] Error: Failed to load Ankh model.")
            return None, None, None
            
        for chain_id, sequence, out_path in tasks:
            embedding = featurize.get_ankh_embed(sequence, f"{pdbid}_{chain_id}", device)
            if embedding is None:
                if verbose: print(f"[rank_lnsk.py] Error: Failed to embed chain {chain_id}")
                continue 
            np.save(out_path, embedding)
        
        del model_ankh, tokenizer_ankh
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    try:
        with open(pocket_path, 'r') as f:
            lines = f.readlines()
        
        res_ids = {
            f"{l[17:20].strip()}{l[21].strip()}{int(l[22:26].strip()):04d}" 
            for l in lines 
            if l.startswith('ATOM') and l[17:20].strip() in featurize.VALID_RES
        }
        
        valid_resdef_list = sorted(list(res_ids))
        
        if len(valid_resdef_list) < featurize.MIN_NUM_PTS:
            if verbose: print(f"[rank_lnsk.py] Error: Not enough residues in pocket: {len(valid_resdef_list)}")
            return None, None, None

    except Exception as e:
        if verbose: print(f"[rank_lnsk.py] Error: Failed to parse pocket file text: {e}")
        return None, None, None

    protein_structure = parse_pdb(protein_path) 
    if protein_structure is None:
        if verbose: print(f"[rank_lnsk.py] Error: Failed to parse protein PDB: {protein_path}")
        return None, None, None

    ftr_data = featurize.featurize(
        ankh_path=ankh_path_dir, 
        protein_structure=protein_structure,
        valid_resdef=valid_resdef_list, 
        pocket_pdb_path=pocket_path,
        protein_pdb_path=protein_path
    )
    
    if ftr_data is None:
        if verbose: print(f"[rank_lnsk.py] Error: Featurization failed for {pocket_path}")
        return None, None, None
    
    node_feats, rots, _, trans, _, _ = ftr_data
    
    N_max = config.NUM_PTS
    n_res = len(node_feats)
    
    if n_res >= N_max:
        poc_idx = np.arange(N_max) 
        mask_np = np.ones(N_max, dtype=bool)
    else:
        poc_idx = np.arange(n_res); poc_pad = N_max - n_res
        poc_idx = np.pad(poc_idx, (0, poc_pad), mode='wrap')
        mask_np = np.ones(N_max, dtype=bool); mask_np[-poc_pad:] = False
    
    h_np = node_feats[poc_idx]; T_np = trans[poc_idx]; R_np = rots[poc_idx]

    h = torch.from_numpy(h_np).float().unsqueeze(0).to(device)
    T = torch.from_numpy(T_np).float().unsqueeze(0).to(device)
    R = torch.from_numpy(R_np).float().unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
    mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
    e, knn_mask = get_edges(T, R, mask_2d) 

    with torch.no_grad():
        h_adapted = model.feature_adaptor(h)
        mu_pernode, _, _ = model.encode(h_adapted, e, mask, knn_mask) 
        
    return mu_pernode, mask, knn_mask


@torch.no_grad()
def l2_cd(mu_A, mask_A, mu_B, mask_B, eps=1e-6):
    B_B, N_A = mu_B.shape[0], mu_A.shape[1]
    
    mu_A_rep = mu_A.expand(B_B, N_A, -1)
    
    dist_matrix_sq = torch.cdist(mu_A_rep, mu_B, p=2)**2
    
    mask_A_float = mask_A.float()
    mask_B_float = mask_B.float()
    
    inf_mask_B = (1.0 - mask_B_float).unsqueeze(1) * 1e9
    dist_matrix_masked_for_A = dist_matrix_sq + inf_mask_B
    min_dist_A_to_B = dist_matrix_masked_for_A.min(dim=2).values
    
    inf_mask_A = (1.0 - mask_A_float).unsqueeze(2) * 1e9
    dist_matrix_masked_for_B = dist_matrix_sq + inf_mask_A
    min_dist_B_to_A = dist_matrix_masked_for_B.min(dim=1).values

    num_valid_A = mask_A_float.sum(dim=1).clamp(min=eps)
    sum_A_to_B = (min_dist_A_to_B * mask_A_float).sum(dim=1)
    mean_A_to_B = sum_A_to_B / num_valid_A
    
    num_valid_B = mask_B_float.sum(dim=1).clamp(min=eps)
    sum_B_to_A = (min_dist_B_to_A * mask_B_float).sum(dim=1)
    mean_B_to_A = sum_B_to_A / num_valid_B
    
    return mean_A_to_B + mean_B_to_A


def main():
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"[rank_lnsk.py]")
        print(f"Loading model and database...")
        print(f"Model Path: {MODEL_PATH}")
        print(f"DB Path   : {DB_PATH}")
        print(f"Batch Size: {COMP_BATCH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[rank_lnsk.py] Error: Model file not found: {MODEL_PATH}")
        sys.exit(1)
        
    model = load_model(MODEL_PATH, device)
    
    if not os.path.exists(DB_PATH):
        if accelerator.is_main_process: print(f"[rank_lnsk.py] Error: DB file not found: {DB_PATH}")
        sys.exit(1)
    
    if accelerator.is_main_process:
        print(f"[rank_lnsk.py] Loading DB pickle...")
        
    with open(DB_PATH, 'rb') as f: 
        latent_db = pickle.load(f)
    
    all_mu_list = [x[0] for x in latent_db]
    all_mask_list = [x[1] for x in latent_db]
    db_paths = [x[2] for x in latent_db]
    
    del latent_db
    gc.collect()
    
    if accelerator.is_main_process:
        print(f"[rank_lnsk.py] Loaded {len(db_paths)} DB entries.")
        os.makedirs(CSV_DIR, exist_ok=True)
        
    query_paths = sorted(glob.glob(os.path.join(QUERY_DIR, "*", "*pocket*.pdb")))
    if not query_paths:
        print(f"[rank_lnsk.py] Error: No query files found in {QUERY_DIR}")
        return

    with accelerator.split_between_processes(query_paths) as rank_paths:
        pass

    if accelerator.is_main_process:
        pbar = tqdm(rank_paths, desc="Processing Queries")
    else:
        pbar = rank_paths
        
    for query_path in pbar:
        try:
            q_dir = os.path.dirname(query_path)
            q_id = os.path.basename(q_dir)
            
            q_filename = os.path.basename(query_path).replace('.pdb', '')
            
            protein_path = os.path.join(q_dir, f"{q_id}_protein.pdb")
            if not os.path.exists(protein_path):
                protein_path = os.path.join(q_dir, "protein.pdb")
            
            if accelerator.is_main_process: pbar.set_postfix({'Query': q_id})

            q_mu, q_mask, _ = encode_query(
                model, protein_path, query_path, device, ANKH_PATH, verbose=True
            )
            
            if q_mu is None: 
                if accelerator.is_main_process: pbar.write(f"[Skip] {q_id}")
                continue
            
            all_dists = []
            
            total_db = len(db_paths)
            for i in range(0, total_db, COMP_BATCH):
                end_i = min(i + COMP_BATCH, total_db)
                
                batch_mu_list = all_mu_list[i:end_i]
                batch_mask_list = all_mask_list[i:end_i]
                
                b_mu = torch.from_numpy(np.vstack(batch_mu_list)).float().to(device)
                b_mask = torch.from_numpy(np.vstack(batch_mask_list)).to(device)
                
                d = l2_cd(q_mu, q_mask, b_mu, b_mask)
                
                all_dists.append(d.cpu())
                
                del b_mu, b_mask, d
            
            dist_np = torch.cat(all_dists).numpy()
            
            df = pd.DataFrame({'Path': db_paths, 'L2_CD_Dist': dist_np})
            df['L2_CD_Rank'] = df['L2_CD_Dist'].rank().astype(int)
            
            out_file = os.path.join(CSV_DIR, f"ranking_{q_id}_{q_filename}.csv")
            
            df.to_csv(out_file, index=False)
            
            del all_dists, dist_np, df, q_mu, q_mask
            
        except Exception as e:
            if accelerator.is_main_process:
                print(f"\n[rank_lnsk.py] Error processing {query_path}: {e}")
            continue

if __name__ == "__main__":
    main()