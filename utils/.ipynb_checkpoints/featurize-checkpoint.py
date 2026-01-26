import os
import sys
import numpy as np
import re
from collections import defaultdict

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import config
    ankh_dim = getattr(config, 'ANKH_DIM', 768)
except ImportError:
    class config:
        ankh_dim = 768

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, T5EncoderModel

try:
    from utils.exclusion import Crystallization_aids, Ligand_exclusion, glycans
    CRYSTALLIZATION_AIDS_SET = {x.strip() for x in Crystallization_aids.split(",")}
    LIGAND_EXCLUSION_SET = {x.strip() for x in Ligand_exclusion.split(",")}
    GLYCANS_SET = {x.strip() for x in glycans.split(",")}
    REMOVE_HET_NAMES = CRYSTALLIZATION_AIDS_SET.union(LIGAND_EXCLUSION_SET).union(GLYCANS_SET)
except ImportError:
    REMOVE_HET_NAMES = set(["SO4", "GOL", "EDO", "HOH"])

VALID_RES = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
             'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}

MIN_NUM_PTS = 10

_ankh_model = None
_ankh_tokenizer = None

def download_ankh():
    ankh_model_dir = "./ankh-base"
    
    bin_path = os.path.join(ankh_model_dir, "pytorch_model.bin")
    config_path = os.path.join(ankh_model_dir, "config.json")

    if os.path.exists(bin_path) and os.path.exists(config_path):
        return ankh_model_dir
    
    if 'HF_MIRROR' in os.environ:
        endpoint = os.environ['HF_MIRROR']
    else:
        endpoint = "https://huggingface.co/"
    
    try:
        print(f"[featurize.py] Ankh model not found in ./ankh-base/pytorch_model.bin. Downloading...")
        snapshot_download(
            repo_id="ElnaggarLab/ankh-base",
            local_dir=ankh_model_dir,
            ignore_patterns=["*.safetensors*"],
            endpoint=endpoint
        )
        print(f"[featurize.py] Ankh model downloaded.")
        return ankh_model_dir
    except Exception as e:
        print(f"[featurize.py] Ankh model download failed: {e}")
        return None


def load_ankh(device):
    global _ankh_model, _ankh_tokenizer
    
    if _ankh_model is not None:
        return _ankh_model, _ankh_tokenizer
    
    ankh_model_dir = download_ankh()
    if ankh_model_dir is None:
        return None, None
    
    try:
        _ankh_tokenizer = AutoTokenizer.from_pretrained(ankh_model_dir)
        
        _ankh_model = T5EncoderModel.from_pretrained(
            ankh_model_dir,
            use_safetensors=False
        )
            
    except Exception as e:
        if "torch.load" in str(e) or "CVE" in str(e):
            print("[utils.featurize] 'torch.load' CVE 감지. trust_remote_code=True로 재시도합니다.")
            try:
                _ankh_model = T5EncoderModel.from_pretrained(
                    ankh_model_dir, 
                    trust_remote_code=True,
                    use_safetensors=False
                )
            except Exception as e2:
                print(f"[featurize.py] Ankh model fallback failed: {e2}")
                import traceback
                traceback.print_exc()
                return None, None
        else:
            print(f"[featurize.py] Ankh model load failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        
    _ankh_model.to(device)
    _ankh_model.eval()
    return _ankh_model, _ankh_tokenizer


def get_ankh_embed(sequence, pocket_id, device):
    if not TORCH_AVAILABLE:
        return None

    model, tokenizer = load_ankh(device)
    if model is None or tokenizer is None:
        return None
    
    try:
        seq_list = list(sequence)
        
        ids = tokenizer.batch_encode_plus(
            [seq_list], 
            add_special_tokens=True, 
            padding=True, 
            is_split_into_words=True, 
            return_tensors="pt"
        )
        
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
            emb = embedding_repr.last_hidden_state[0, :len(sequence)].cpu().numpy()
        
        del input_ids, attention_mask, embedding_repr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return emb
        
    except Exception as e:
        print(f"[featurize.py] get_ankh_embed failed: {e}")
        return None

def get_seq_map(pdb_path):
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    mapping_per_chain = defaultdict(dict)
    
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
                chain_residues[chain_id].append((res_num, res_name, chain_id))
        
        sorted_chain_ids = sorted(chain_residues.keys())
        
        for chain_id in sorted_chain_ids:
            chain_res_list = sorted(chain_residues[chain_id], key=lambda x: x[0])
            
            chain_local_idx = 0
            for res_num, res_name, chain_id_char in chain_res_list:
                resdef_id = f"{res_name}{chain_id_char}{res_num:04d}"
                mapping_per_chain[chain_id][resdef_id] = chain_local_idx
                chain_local_idx += 1
    
    except Exception as e:
        print(f"[featurize.py] get_seq_map failed: {e}")
        return {}
    
    return mapping_per_chain


def load_embed(ankh_path_dir, pdb_path, valid_resdef):
    pdbid = os.path.basename(os.path.dirname(pdb_path))

    try:
        loaded_chain_embeddings = {}
        required_chains = set()
        for rdef in valid_resdef:
            try:
                chainID_char = rdef[3]
                required_chains.add(chainID_char)
            except IndexError:
                continue 
        
        if not required_chains:
            print(f"[featurize.py] No required chains derived from valid_resdef.")
            return None

        residue_mapping_per_chain = get_seq_map(pdb_path)
        if not residue_mapping_per_chain:
            print(f"[featurize.py] Sequence mapping failed for {pdb_path}")
            return None

        for chain_id in required_chains:
            chain_ankh_path = os.path.join(ankh_path_dir, f"{pdbid}_chain_{chain_id}.npy")
            
            if not os.path.exists(chain_ankh_path):
                print(f"[featurize.py] Missing embedding file: {chain_ankh_path}")
                return None
            
            loaded_chain_embeddings[chain_id] = np.load(chain_ankh_path)
            
            if chain_id in residue_mapping_per_chain:
                map_len = len(residue_mapping_per_chain[chain_id])
                emb_len = loaded_chain_embeddings[chain_id].shape[0]
                if map_len != emb_len:
                    print(f"[featurize.py] Length mismatch for {pdbid} Chain {chain_id}: "
                          f"SeqMap detected {map_len} residues vs Embedding file has {emb_len}.")
                    return None
            else:
                print(f"[featurize.py] Chain {chain_id} found in resdef but not in PDB sequence map.")
                return None

        filtered_emb = []
        for rdef in valid_resdef:
            try:
                chain_id = rdef[3]
                if rdef not in residue_mapping_per_chain[chain_id]:
                    print(f"[featurize.py] Residue {rdef} (Chain {chain_id}) in pocket but NOT found in PDB sequence map. (Check parsing/filtering)")
                    return None

                idx = residue_mapping_per_chain[chain_id][rdef]
                chain_emb = loaded_chain_embeddings[chain_id]
                
                if idx < chain_emb.shape[0]:
                    filtered_emb.append(chain_emb[idx])
                else:
                    print(f"[featurize.py] Index {idx} out of bounds for chain {chain_id} (len {chain_emb.shape[0]})")
                    return None
            except (KeyError, IndexError) as e:
                print(f"[featurize.py] Exception during embedding extraction: {e}")
                return None
        
        if len(filtered_emb) != len(valid_resdef):
            return None
        
        result = np.array(filtered_emb, dtype=np.float32)
        del loaded_chain_embeddings, residue_mapping_per_chain, filtered_emb
        
        return result
        
    except Exception as e:
        print(f"[featurize.py] load_embed Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_frames(coords_n, coords_ca, coords_c):
    T = coords_ca
    v1 = coords_c - coords_ca
    
    if TORCH_AVAILABLE:
        try:
            with torch.no_grad():
                v1_torch = torch.from_numpy(v1).float()
                e1 = F.normalize(v1_torch, dim=-1)
                
                v2 = coords_n - coords_ca
                v2_torch = torch.from_numpy(v2).float()
                
                u2 = v2_torch - torch.sum(v2_torch * e1, dim=-1, keepdim=True) * e1
                e2 = F.normalize(u2, dim=-1)
                e3 = torch.cross(e1, e2, dim=-1)
                R = torch.stack([e1, e2, e3], dim=2)
                R_numpy = R.cpu().numpy()
            
            del v1_torch, e1, v2_torch, u2, e2, e3, R
            
            return T, R_numpy
        except Exception:
            pass
    
    e1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    
    v2 = coords_n - coords_ca
    proj = np.sum(v2 * e1, axis=-1, keepdims=True) * e1
    u2 = v2 - proj
    e2 = u2 / (np.linalg.norm(u2, axis=-1, keepdims=True) + 1e-8)
    
    e3 = np.cross(e1, e2, axis=-1)
    
    R_numpy = np.stack([e1, e2, e3], axis=2)
    
    return T, R_numpy


def normalize_coords(coords, target_max_dist=1.0):
    mean_coord = np.mean(coords, axis=0)
    zero_centered = coords - mean_coord
    
    dist_from_origin = np.linalg.norm(zero_centered, axis=1)
    max_dist = np.max(dist_from_origin)
    
    if max_dist < 1e-6:
        scaling_factor = 1.0
    else:
        scaling_factor = max_dist / target_max_dist
    
    scaled_coords = zero_centered / scaling_factor
    
    return scaled_coords, scaling_factor, mean_coord


def denormalize_coords(scaled_coords, scaling_factor, mean_coord):
    denormalized = scaled_coords * scaling_factor
    original_coords = denormalized + mean_coord
    return original_coords


def get_sc_vec(res_dict, R_local):
    sc_atoms = []
    
    if 'atoms' not in res_dict:
        return np.zeros(3, dtype=np.float32)
        
    for atom_id, coord in res_dict['atoms'].items():
        if atom_id not in ['N', 'CA', 'C', 'O']:
            sc_atoms.append(coord)
    
    if not sc_atoms:
        return np.zeros(3, dtype=np.float32)
    
    sc_coords = np.array(sc_atoms, dtype=np.float32)
    centroid = np.mean(sc_coords, axis=0)
    
    if 'CA' not in res_dict['atoms']:
        return np.zeros(3, dtype=np.float32)
        
    ca_coord = res_dict['atoms']['CA']
    
    vec_global = centroid - ca_coord
    
    vec_local = R_local.T @ vec_global
    
    dist = np.linalg.norm(vec_local)
    if dist < 1e-6:
        return np.zeros(3, dtype=np.float32)
    
    unit_vec_local = vec_local / dist
    
    return unit_vec_local.astype(np.float32)

RES_NUM_EMBED_SIZE = 2000
CHAIN_ID_EMBED_SIZE = 32
_CHAIN_MAP_A_Z = {chr(65+i): i for i in range(26)}
_CHAIN_OTHER_START_IDX = 26

def _get_chain_int(chain_id):
    if chain_id in _CHAIN_MAP_A_Z:
        return _CHAIN_MAP_A_Z[chain_id]
    
    other_val = ord(chain_id) % (CHAIN_ID_EMBED_SIZE - _CHAIN_OTHER_START_IDX)
    return _CHAIN_OTHER_START_IDX + other_val


def featurize(ankh_path, protein_structure, valid_resdef, 
              pocket_pdb_path=None, protein_pdb_path=None):
    try:
        if protein_structure is None:
            return None
        
        num_residues = len(valid_resdef)

        if os.path.isdir(ankh_path):
            ankh_path_dir = ankh_path
        else:
            ankh_path_dir = os.path.dirname(ankh_path)
        
        pocket_ankh_embeddings = load_embed(
            ankh_path_dir, 
            protein_pdb_path, 
            valid_resdef
        )
        
        if pocket_ankh_embeddings is None:
            return None
        
        if pocket_ankh_embeddings.shape[1] != ankh_dim:
            print(f"[featurize.py] Embedding dim mismatch: Got {pocket_ankh_embeddings.shape[1]}, expected {ankh_dim}")
            return None
        if pocket_ankh_embeddings.shape[0] != num_residues:
            print(f"[featurize.py] Embedding count mismatch: Got {pocket_ankh_embeddings.shape[0]}, expected {num_residues}")
            return None

        node_feats = np.empty((num_residues, ankh_dim), dtype=np.float32)
        coords_n = np.empty((num_residues, 3), dtype=np.float32)
        coords_ca = np.empty((num_residues, 3), dtype=np.float32)
        coords_c = np.empty((num_residues, 3), dtype=np.float32)
        
        res_nums = np.empty((num_residues,), dtype=np.int32)
        chain_ids = np.empty((num_residues,), dtype=np.int32)
        
        processed_count = 0

        for i, rdef in enumerate(valid_resdef):
            resName = rdef[:3]
            chainID_char = rdef[3]
            resNum_int = int(rdef[4:])

            try:
                res_dict = protein_structure[chainID_char][resNum_int]
            except KeyError:
                print(f"[featurize.py] Residue {rdef} not found in protein structure dict.")
                return None

            node_feats[i] = pocket_ankh_embeddings[i]
            
            try:
                coords_n[i] = res_dict['atoms']['N']
                coords_ca[i] = res_dict['atoms']['CA']
                coords_c[i] = res_dict['atoms']['C']
            except KeyError as e:
                print(f"[featurize.py] Missing backbone atom {e} in residue {rdef}")
                return None
            
            res_nums[i] = np.clip(resNum_int, 0, RES_NUM_EMBED_SIZE - 1) 
            chain_ids[i] = _get_chain_int(chainID_char)

            processed_count += 1

        del pocket_ankh_embeddings

    except Exception as e:
        import sys
        if not hasattr(featurize, '_error_count'):
            featurize._error_count = 0
        featurize._error_count += 1
        if featurize._error_count <= 5:
            print(f"\n[FEATURIZE ERROR #{featurize._error_count}]: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        return None

    if processed_count < MIN_NUM_PTS:
        print(f"[featurize.py] Not enough valid points processed: {processed_count} < {MIN_NUM_PTS}")
        return None

    try:
        translations, rotations = get_frames(coords_n, coords_ca, coords_c)
        del coords_n, coords_c
    except Exception as e:
        print(f"[featurize.py] get_frames failed: {e}")
        return None

    sc_vecs = np.empty((num_residues, 3), dtype=np.float32)
    
    for i, rdef in enumerate(valid_resdef):
        resName = rdef[:3]
        chainID = rdef[3]
        resNum = int(rdef[4:])
        
        try:
            res_dict = protein_structure[chainID][resNum]
            R_local = rotations[i]
            
            sc_vecs[i] = get_sc_vec(
                res_dict, R_local
            )
        except Exception as e:
            sc_vecs[i] = np.zeros(3, dtype=np.float32)

    return node_feats, rotations, sc_vecs, translations, res_nums, chain_ids


def valid_ligand(ligand_name):
    if ligand_name in REMOVE_HET_NAMES:
        return False
    if re.search(r'^\d', ligand_name):
        return False
    if len(ligand_name) < 2:
        return False
    return True