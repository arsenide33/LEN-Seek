import os
import sys
import glob
import numpy as np
import torch
import gc
import argparse
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import config
from utils.featurize import load_ankh, get_ankh_embed

def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    is_ddp = world_size > 1
    
    if is_ddp:
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        if is_ddp:
            print(f"[embed.py] Running with Multi-GPU({world_size})")
        else:
            print(f"[embed.py] Running single GPU/CPU mode.")
        print(f"Device: {device}\n")
    
    return rank, world_size, device, is_ddp

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_chain_sequences(pdb_path):
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    chain_sequences = {}
    
    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        chain_residues = defaultdict(list)
        seen_residues = set()

        for line in lines:
            if not line.startswith('ATOM'):
                continue
            
            res_name = line[17:20].strip()
            if res_name not in three_to_one:
                continue
                
            chain_id = line[21].strip()
            if not chain_id:
                continue
                
            try:
                res_num = int(line[22:26].strip())
            except:
                continue
            
            key = (chain_id, res_num)
            if key not in seen_residues:
                seen_residues.add(key)
                chain_residues[chain_id].append((res_num, res_name))
        
        sorted_chain_ids = sorted(chain_residues.keys())
        
        for chain_id in sorted_chain_ids:
            chain_res_list = sorted(chain_residues[chain_id], key=lambda x: x[0])
            
            sequence_parts = []
            for res_num, res_name in chain_res_list:
                sequence_parts.append(three_to_one.get(res_name, 'X'))

            sequence = ''.join(sequence_parts)
            
            if sequence:
                chain_sequences[chain_id] = sequence
                
        return chain_sequences
            
    except Exception as e:
        print(f"[embed.py] Error parsing {pdb_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Generate Ankh Embeddings")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'],
                        help="Mode to run: 'train' or 'eval'")
    args = parser.parse_args()

    rank, world_size, device, is_ddp = setup_ddp()

    if args.mode == 'train':
        db_path = config.TRAIN_DB_PATH
        out_dir = config.TRAIN_ANKH_PATH
    else:
        db_path = config.EVAL_DB_PATH
        out_dir = config.EVAL_ANKH_PATH
    
    os.makedirs(out_dir, exist_ok=True)
    
    all_protein_files = sorted(glob.glob(f'{db_path}/*/protein.pdb'))
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[embed.py]")
        print(f"Ankh embedding pre-computation")
        print(f"{'='*60}")
        print(f"Mode: {args.mode.upper()}")
        print(f"Database path: {db_path}")
        print(f"Output directory: {out_dir}")
        print(f"Found total {len(all_protein_files)} protein files")
        print(f"{'='*60}\n")
    
    if not all_protein_files:
        if rank == 0:
            print(f"[embed.py] Error: No protein.pdb files found in {db_path}")
        cleanup_ddp()
        return

    rank_files = all_protein_files[rank::world_size]

    if rank == 0:
        print(f"[embed.py] Loading Ankh model (Rank {rank}, Device {device})...")
    
    model, tokenizer = load_ankh(device)

    if model is None:
        if rank == 0:
            print(f"[embed.py] Error: Failed to load Ankh model")
        cleanup_ddp()
        return
    
    if rank == 0:
        print(f"[embed.py] Ankh model loaded. Rank {rank} processing {len(rank_files)} files.\n")
    
    if is_ddp:
        dist.barrier()
        
    success_count = 0
    skip_count = 0
    error_count = 0
    
    if rank == 0:
        pbar = tqdm(enumerate(rank_files), desc=f"[GPU {rank}] Generating", total=len(rank_files))
    else:
        pbar = enumerate(rank_files)

    for idx, protein_path in pbar:
        try:
            pdbid = os.path.basename(os.path.dirname(protein_path))
            
            sequences_dict = get_chain_sequences(protein_path)
            
            if not sequences_dict:
                error_count += 1
                if rank == 0:
                    pbar.write(f"\n[Warning] No valid sequences found for {pdbid}. Skipping.")
                continue

            for chain_id, sequence in sequences_dict.items():
                output_path = os.path.join(out_dir, f"{pdbid}_chain_{chain_id}.npy")
            
                if os.path.exists(output_path):
                    skip_count += 1
                    continue
                
                if sequence is None or len(sequence) == 0:
                    error_count += 1
                    continue
                
                embedding = get_ankh_embed(sequence, f"{pdbid}_chain_{chain_id}", device)
                if embedding is None:
                    error_count += 1
                    if rank == 0:
                        pbar.write(f"\nError: Failed to generate embedding for {pdbid}_chain_{chain_id}")
                    continue
                
                np.save(output_path, embedding)
                success_count += 1
                
                del sequence, embedding
            
            if (idx + 1) % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            error_count += 1
            if rank == 0:
                pbar.write(f"\nError: Exception processing {protein_path}: {e}")
            continue
    
    if is_ddp:
        dist.barrier()

    if rank == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print(f"[embed.py]")
        print(f"Embedding Generation Complete. ({args.mode.upper()} Mode)")
        print(f"{'='*60}")
        print(f"Total protein files processed: {len(all_protein_files)}")
        
        all_files_npy = glob.glob(f'{out_dir}/*.npy')
        print(f"Total .npy files (per chain) in output directory: {len(all_files_npy)}")
        print(f"  (Success: {success_count}, Skipped: {skip_count}, Errors: {error_count})")
        print(f"{'='*60}\n")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()