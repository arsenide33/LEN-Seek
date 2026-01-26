import os
import glob
import shutil
import subprocess
from collections import defaultdict
from tqdm import tqdm

MMSEQS_BIN = '/home/khyeo/Project/bsVAE/mmseqs/bin/mmseqs'

DB_PATH_FULL = '/home/khyeo/DB/CASF-2016/coreset'
DB_PATH_EX = '/home/khyeo/DB/PDBbind'
REDUNDANT_DIR = '/home/khyeo/DB/PDBbind_redn'

SIMILARITY_THRESHOLD = 0.8

def get_chain_sequences(pdb_path):
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
    except: return {}

    chain_residues = defaultdict(list)
    seen = set()

    for line in lines:
        if not line.startswith('ATOM'): continue
        res_name = line[17:20].strip()
        if res_name not in three_to_one: continue
        
        raw_chain = line[21].strip()
        chain_id = raw_chain if raw_chain else 'A'
        
        try:
            res_num = int(line[22:26].strip())
            i_code = line[26].strip()
        except: continue
        
        key = (chain_id, res_num, i_code)
        if key not in seen:
            seen.add(key)
            chain_residues[chain_id].append((key, res_name))

    chain_seqs = {}
    for c in sorted(chain_residues.keys()):
        residues = sorted(chain_residues[c], key=lambda x: (x[0][1], x[0][2]))
        seq = "".join([three_to_one[r[1]] for r in residues])
        if len(seq) > 10:
            chain_seqs[c] = seq
    return chain_seqs

def create_chain_fasta_from_db(db_path, output_fasta, desc_text):
    pdb_dirs = sorted(glob.glob(os.path.join(db_path, '*')))
    pdb_dirs = [d for d in pdb_dirs if os.path.isdir(d)]
    
    if not pdb_dirs:
        print(f"[Warning] No directories found in {db_path}")
        return False

    print(f"Converting {desc_text} to FASTA...")
    count = 0
    
    with open(output_fasta, 'w') as f_out:
        for pdb_dir in tqdm(pdb_dirs, desc=f"Processing {desc_text}"):
            pdb_id = os.path.basename(pdb_dir)
            
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
            
            if not os.path.exists(pdb_path):
                pdb_path = os.path.join(pdb_dir, "protein.pdb")
            
            if not os.path.exists(pdb_path):
                continue

            chains = get_chain_sequences(pdb_path)
            if chains:
                count += 1
                for c, seq in chains.items():
                    f_out.write(f">{pdb_id}__{c}\n{seq}\n")
    
    print(f" -> Processed {count} PDBs from {desc_text}")
    return True

def main():
    if not os.path.exists(MMSEQS_BIN):
        print(f"[Critical] MMseqs binary not found at {MMSEQS_BIN}"); return

    if not os.path.exists(REDUNDANT_DIR):
        os.makedirs(REDUNDANT_DIR)

    target_fasta = "temp_target_full.fasta"
    query_fasta = "temp_query_ex.fasta"
    res_external = "mmseqs_external.m8"
    res_internal_prefix = "mmseqs_internal"
    tmp_dir = "mmseqs_tmp"

    redundant_pdbs = set()

    try:
        if not create_chain_fasta_from_db(DB_PATH_FULL, target_fasta, "Full DB (Target)"): return
        if not create_chain_fasta_from_db(DB_PATH_EX, query_fasta, "EX DB (Query)"): return

        print(f"\n[Step 1] Checking External Redundancy (Query vs Target)...")
        print(f" -> Similarity Threshold: {SIMILARITY_THRESHOLD}")
        
        cmd_ext = [
            MMSEQS_BIN, "easy-search",
            query_fasta, target_fasta, res_external, tmp_dir,
            "--min-seq-id", str(SIMILARITY_THRESHOLD),
            "--format-output", "query,target,pident",
            "-c", "0.8", "--cov-mode", "2"
        ]
        subprocess.run(cmd_ext, check=True)
        
        ext_count = 0
        if os.path.exists(res_external):
            with open(res_external, 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts: continue
                    
                    q_header = parts[0]
                    t_header = parts[1]
                    
                    q_pdb = q_header.rsplit("__", 1)[0]
                    t_pdb = t_header.rsplit("__", 1)[0]
                    
                    redundant_pdbs.add(q_pdb)
                    ext_count += 1
                    
        print(f" -> Found {len(redundant_pdbs)} PDBs overlapping with Full DB.")
        print(f"\n[Step 2] Checking Internal Redundancy (Query vs Query)...")
        cmd_int = [
            MMSEQS_BIN, "easy-cluster",
            query_fasta, res_internal_prefix, tmp_dir,
            "--min-seq-id", str(SIMILARITY_THRESHOLD),
            "-c", "0.8", "--cov-mode", "2"
        ]
        subprocess.run(cmd_int, check=True)

        cluster_tsv = f"{res_internal_prefix}_cluster.tsv"
        int_count = 0
        
        if os.path.exists(cluster_tsv):
            with open(cluster_tsv, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    
                    rep_header = parts[0]
                    mem_header = parts[1]
                    
                    if rep_header == mem_header: continue
                    
                    rep_pdb = rep_header.rsplit("__", 1)[0]
                    mem_pdb = mem_header.rsplit("__", 1)[0]
                    
                    if rep_pdb == mem_pdb: continue
                    
                    if mem_pdb not in redundant_pdbs:
                        redundant_pdbs.add(mem_pdb)
                        int_count += 1
                        
        print(f" -> Found {int_count} additional PDBs overlapping internally.")
        print(f"\n{'='*70}")
        print(f"Final Filtering Result (Threshold: {SIMILARITY_THRESHOLD})")
        print(f"{'='*70}")
        print(f"Total Redundant PDBs to move: {len(redundant_pdbs)}")

        moved_count = 0
        for pdb_id in tqdm(list(redundant_pdbs), desc="Moving Folders"):
            src_path = os.path.join(DB_PATH_EX, pdb_id)
            dst_path = os.path.join(REDUNDANT_DIR, pdb_id)
            
            if os.path.exists(src_path):
                try:
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                except Exception as e:
                    print(f"[Error] Failed to move {pdb_id}: {e}")
        
        print(f"\nSuccessfully moved {moved_count} folders to:")
        print(f"  -> {REDUNDANT_DIR}")

    except Exception as e:
        print(f"[Error] {e}")
    finally:
        if os.path.exists(target_fasta): os.remove(target_fasta)
        if os.path.exists(query_fasta): os.remove(query_fasta)
        if os.path.exists(res_external): os.remove(res_external)
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
        for f in glob.glob(f"{res_internal_prefix}*"):
            os.remove(f)

if __name__ == "__main__":
    main()