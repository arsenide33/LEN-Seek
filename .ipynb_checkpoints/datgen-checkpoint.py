import os
import pickle
import sys
import gc
import argparse

import config
from utils.bsdb import PDBDataParallel

def create_dataset():
    parser = argparse.ArgumentParser(description="Generate PDB Dataset Pickle")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'],
                        help="Mode to run: 'train' or 'eval'")
    args = parser.parse_args()

    if args.mode == 'train':
        dataset_path = config.TRAIN_DATA_PATH
        db_path = config.TRAIN_DB_PATH
        ankh_path = config.TRAIN_ANKH_PATH
    else:  # eval
        dataset_path = config.EVAL_DATA_PATH
        db_path = config.EVAL_DB_PATH
        ankh_path = config.EVAL_ANKH_PATH
    print(f"\n{'='*60}")
    print(f"[datgen.py]")
    print(f"Dataset Generation Mode: {args.mode.upper()}")
    print(f"Source DB Path : {db_path}")
    print(f"Ankh Embed Path: {ankh_path}")
    print(f"Output Pickle  : {dataset_path}")
    print(f"{'='*60}\n")
    
    if os.path.exists(dataset_path):
        print(f"[datgen.py] Dataset file already exists: {dataset_path}")
        print("Delete this file first if you want to regenerate.")
        sys.exit(0)
        
    print(f"[datgen.py] Dataset not found. Creating...")
    
    try:
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        dataset = PDBDataParallel(
            pocPts=config.NUM_PTS,
            DBPath=db_path,
            invalid_site_path=config.INVALID_PATH,
            ankh_path=ankh_path
        )
        
        if len(dataset) == 0:
            print(f"\n[datgen.py] Error: Dataset creation resulted in 0 valid sites.")
            sys.exit(1)

        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n[dagten.py] Dataset saved successfully to {dataset_path} (Items: {len(dataset)})")

    except Exception as e:
        print(f"\n[datgen.py] Error occurred during dataset generation:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        gc.collect()

if __name__ == "__main__":
    create_dataset()