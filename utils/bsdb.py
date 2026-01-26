import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import glob
import gc
import multiprocessing as mp
from functools import partial
from collections import defaultdict
import re

import config 

from utils.featurize import (
    featurize, 
    valid_ligand, VALID_RES, MIN_NUM_PTS
)
from utils.pdb_parser import parse_pdb

def _get_chain_ranges(protein_pdb_path):
    chain_ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
    seen_residues = set()
    try:
        with open(protein_pdb_path, 'r') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                
                chain_id = line[21].strip()
                try:
                    res_num = int(line[22:26].strip())
                except:
                    continue

                key = (chain_id, res_num)
                if key not in seen_residues:
                    seen_residues.add(key)
                    if res_num < chain_ranges[chain_id]['min']:
                        chain_ranges[chain_id]['min'] = res_num
                    if res_num > chain_ranges[chain_id]['max']:
                        chain_ranges[chain_id]['max'] = res_num
    except Exception:
        return {}
    return chain_ranges


class PDBData(Dataset):
    def _load_invalid(self, path, db_path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        print(f"\n'{path}' not found. Scanning DB to create it...")
        pocket_files = glob.glob(f'{db_path}/*/ligand_pocket*.pdb')
        invalid_sites = []
        for p_path in tqdm(pocket_files, desc="Scanning invalid pockets"):
            try:
                with open(p_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if line.startswith('ATOM') and line[17:20].strip() not in VALID_RES:
                        invalid_sites.append(p_path)
                        break
            except Exception:
                invalid_sites.append(p_path) 
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(path, 'wb') as f:
            pickle.dump(invalid_sites, f)
        print(f"Invalid site list saved ({len(invalid_sites)} sites).")
        return invalid_sites

    def __init__(self, pocPts, DBPath, invalid_site_path, ankh_path):
        self.pocPts = pocPts
        self.path = []
        self.data = []
        db_path = DBPath
        self.ankh_path = ankh_path

        print(f"\n{'='*70}\nInitializing PDBData (Serial)")
        print(f"Database path: {db_path}")
        print(f"Ankh embeddings dir: {self.ankh_path}")
        print(f"{'='*70}\n")
        
        invalid_sites_set = set(self._load_invalid(invalid_site_path, db_path))
        pocket_files = sorted(glob.glob(f'{db_path}/*/ligand_pocket*.pdb'))
        print(f"Found {len(pocket_files)} pocket files")
        print(f"Invalid sites to skip: {len(invalid_sites_set)}\n")
        
        processed_count = 0
        success_count = 0
        error_counts = defaultdict(int)
        batch_size = 500 
        temp_data = []
        temp_paths = []

        for idx, p_path in enumerate(tqdm(pocket_files, desc="Processing pockets (Serial)")):
            processed_count += 1
            if p_path in invalid_sites_set:
                error_counts['invalid_site'] += 1
                continue
            protein_path = os.path.join(os.path.dirname(p_path), 'protein.pdb')
            if not os.path.exists(protein_path):
                error_counts['no_protein'] += 1
                continue
            try:
                pdbid = os.path.basename(os.path.dirname(p_path))

                search_pattern = os.path.join(self.ankh_path, f"{pdbid}_chain_*.npy")
                found_embeddings = glob.glob(search_pattern)
                
                if not found_embeddings:
                    error_counts['no_ankh_embed'] += 1
                    if error_counts['no_ankh_embed'] <= 1:
                        tqdm.write(f"[DEBUG] No embedding files found for {pdbid} in {self.ankh_path}")
                    continue

                with open(p_path, 'r') as f:
                    lines = f.readlines()
                res_ids = {f"{l[17:20].strip()}{l[21].strip()}{int(l[22:26].strip()):04d}" for l in lines if l.startswith('ATOM') and l[17:20].strip() in VALID_RES}
                del lines 
                if len(res_ids) < MIN_NUM_PTS:
                    error_counts['min_pts'] += 1
                    continue

                chain_ranges = _get_chain_ranges(protein_path)
                if not chain_ranges:
                    error_counts['protein_parse_fail'] += 1
                    continue

                ligand_file = p_path.replace('ligand_pocket_', 'ligand_')
                ligand_name = None
                if os.path.exists(ligand_file):
                    try:
                        with open(ligand_file, 'r') as f:
                            for line in f:
                                if line.startswith('HETATM'):
                                    ligand_name = line[17:20].strip()
                                    break
                    except: pass 
                if ligand_name is None or not valid_ligand(ligand_name):
                    error_counts['no_ligand'] += 1
                    continue

                structure_dict = parse_pdb(protein_path)
                if structure_dict is None:
                    error_counts['structure_parse'] += 1
                    continue
                
                valid_resdef_list = sorted(list(res_ids))
                
                ftr = featurize(
                    ankh_path=self.ankh_path,
                    protein_structure=structure_dict,
                    valid_resdef=valid_resdef_list,
                    pocket_pdb_path=p_path,
                    protein_pdb_path=protein_path
                )

                if ftr is not None:
                    temp_data.append(ftr) 
                    temp_paths.append(p_path)
                    success_count += 1
                else:
                    error_counts['featurize_failed'] += 1

                del structure_dict 
                del res_ids

                if (idx + 1) % batch_size == 0:
                    self.data.extend(temp_data)
                    self.path.extend(temp_paths)
                    temp_data = []
                    temp_paths = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                error_counts['unknown_exception'] += 1 
                tqdm.write(f"Unexpected error processing {p_path}: {e}")
                continue

        if temp_data:
            self.data.extend(temp_data)
            self.path.extend(temp_paths)
        del temp_data 
        del temp_paths
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n{'='*70}\nDataset Creation Complete!\n{'='*70}")
        print(f"Total processed pocket files: {processed_count}")
        print(f"Successfully featurized sites: {success_count} (Raw)")
        print(f"\nError breakdown:")
        for error_type, count in error_counts.items():
            if count > 0:
                print(f"  - {error_type}: {count}")
        print(f"{'='*70}\n")


    def __len__(self):
        if not hasattr(self, 'data') or len(self.data) == 0:
            return 0
        return len(self.data)

    def __getitem__(self, idx):
        real_idx = idx % len(self.data)
        
        node_feats, rots, sc_vecs, trans, res_nums, chain_ids = self.data[real_idx]
        num_coords = len(trans)

        if num_coords >= self.pocPts:
            poc_idx = np.random.choice(num_coords, self.pocPts, replace=False)
            mask_np = np.ones(self.pocPts, dtype=bool)
        else:
            poc_idx = np.arange(num_coords)
            poc_pad = self.pocPts - num_coords
            poc_idx = np.pad(poc_idx, (0, poc_pad), mode='wrap')
            mask_np = np.ones(self.pocPts, dtype=bool)
            mask_np[-poc_pad:] = False

        h_np = node_feats[poc_idx]
        r_np = rots[poc_idx] 
        sc_np = sc_vecs[poc_idx]
        t_np = trans[poc_idx]
        res_np = res_nums[poc_idx]
        chain_np = chain_ids[poc_idx]

        h = torch.from_numpy(h_np).float()
        r = torch.from_numpy(r_np).float() 
        sc = torch.from_numpy(sc_np).float()
        t = torch.from_numpy(t_np).float()
        mask = torch.from_numpy(mask_np)
        res_ids = torch.from_numpy(res_np).long()
        chain_ids = torch.from_numpy(chain_np).long()

        return h, sc, t, mask, r, res_ids, chain_ids


def _worker(p_path, db_path, ankh_path):
    try:
        protein_path = os.path.join(os.path.dirname(p_path), 'protein.pdb')
        if not os.path.exists(protein_path):
            return ('no_protein', p_path)

        pdbid = os.path.basename(os.path.dirname(p_path))
        
        search_pattern = os.path.join(ankh_path, f"{pdbid}_chain_*.npy")
        if not glob.glob(search_pattern):
            return ('no_ankh_embed', p_path)

        with open(p_path, 'r') as f:
            lines = f.readlines()
        res_ids = {f"{l[17:20].strip()}{l[21].strip()}{int(l[22:26].strip()):04d}" for l in lines if l.startswith('ATOM') and l[17:20].strip() in VALID_RES}
        del lines
        if len(res_ids) < MIN_NUM_PTS:
            return ('min_pts', p_path)

        chain_ranges = _get_chain_ranges(protein_path)
        if not chain_ranges:
            return ('protein_parse_fail', p_path)

        ligand_file = p_path.replace('ligand_pocket_', 'ligand_')
        ligand_name = None
        if os.path.exists(ligand_file):
            try:
                with open(ligand_file, 'r') as f:
                    for line in f:
                        if line.startswith('HETATM'):
                            ligand_name = line[17:20].strip()
                            break
            except Exception:
                pass 
        if ligand_name is None or not valid_ligand(ligand_name):
            return ('no_ligand', p_path)

        structure_dict = parse_pdb(protein_path)
        if structure_dict is None:
            return ('structure_parse', p_path)
        
        valid_resdef_list = sorted(list(res_ids))
        
        ftr = featurize(
            ankh_path=ankh_path,
            protein_structure=structure_dict,
            valid_resdef=valid_resdef_list,
            pocket_pdb_path=p_path,
            protein_pdb_path=protein_path
        )
        del structure_dict, res_ids

        if ftr is not None:
            return ('success', (ftr, p_path))
        else:
            return ('featurize_failed', p_path)

    except Exception as e:
        return ('exception', f"{p_path}: {e}")

class PDBDataParallel(PDBData):
    
    def __init__(self, pocPts, DBPath, invalid_site_path, ankh_path):
        self.pocPts = pocPts
        self.path = []
        self.data = []
        db_path = DBPath
        self.ankh_path = ankh_path
        
        print(f"\n{'='*70}\nInitializing PDBDataParallel (CPU x{os.cpu_count()})")
        print(f"Database path: {db_path}")
        print(f"Ankh embeddings dir: {self.ankh_path}")
        print(f"{'='*70}\n")
        
        invalid_sites_set = set(self._load_invalid(invalid_site_path, db_path))
        
        pocket_files = sorted(glob.glob(f'{db_path}/*/ligand_pocket*.pdb'))
        print(f"Found {len(pocket_files)} total pocket files.")
        
        valid_pocket_files = [p for p in pocket_files if p not in invalid_sites_set]
        print(f"Skipping {len(invalid_sites_set)} invalid sites (from list).")
        print(f"Processing {len(valid_pocket_files)} valid pocket files...")

        num_cpus = os.cpu_count()
        worker_func = partial(_worker, db_path=db_path, ankh_path=self.ankh_path)
        
        results = []
        error_counts = defaultdict(int)
        success_count = 0
        
        with mp.Pool(processes=num_cpus) as pool:
            progress_bar = tqdm(
                pool.imap(worker_func, valid_pocket_files), 
                total=len(valid_pocket_files), 
                desc="Featurizing (parallel)"
            )
            for result in progress_bar:
                status, value = result
                if status == 'success':
                    ftr, p_path = value
                    self.data.append(ftr)
                    self.path.append(p_path)
                    success_count += 1
                else:
                    error_counts[status] += 1
                
                progress_bar.set_postfix({
                    'Success': success_count, 
                    'Failed': sum(error_counts.values())
                })
        
        gc.collect()

        print(f"\n{'='*70}\nParallel Dataset Creation Complete!\n{'='*70}")
        print(f"Total processed pocket files: {len(valid_pocket_files)}")
        print(f"Successfully featurized sites: {success_count} (Raw)")
        print(f"\nError breakdown (Skipped):")
        if not error_counts:
            print("  - None")
        for error_type, count in error_counts.items():
            print(f"  - {error_type}: {count}")
        print(f"{'='*70}\n")