import os
import torch

TRAIN_DB_PATH = '/home/khyeo/DB/BsitePDB/BSDB_231103'
EVAL_DB_PATH = '/home/khyeo/DB/BsitePDB/BSDB_EX'

TRAIN_ANKH_PATH = f'/home/khyeo/DB/BsitePDB/ankh_train'
EVAL_ANKH_PATH = f'/home/khyeo/DB/BsitePDB/ankh_eval'
CKPT_PATH = f'./ckpt/lenseek_pretrained.pth'


TRAIN_DATA_PATH = f'./db/train_DB_precomputed.pkl' 
EVAL_DATA_PATH = f'./db/eval_DB.pkl' # For test only
LATENT_DATA_PATH = f'./db/latent_DB_precomputed.pkl'

# -----------------------------------------------------------------------------
INVALID_PATH = f'./db/invalid_sites.pkl'
ANKH_DIM = 768
NUM_PTS = 65
EDGE_DIM = 39
NODE_DIM = ANKH_DIM # 768
EPOCHS = 200
BATCH_SIZE = 64
START_LR = 1e-3
END_LR = 1e-4
MAX_GRAD = 1.0
BETA_MAX = 0.5
KL_CYCLES = 1
KL_START = 50
FREE_BITS = 12.0
GNN_H = 512
GNN_Z = 256
GNN_LAYERS = 4
GNN_HEADS = 8
GNN_DROP = 0.1
K_NEIGHBORS = 15
W_KL = 1.0
W_ANKH_MSE = 1.0
W_ANKH_COS = 40.0
W_ANKH_NORM = 4.0
W_FAPE = 20.0
W_RMSD = 5.0
LOGVAR_MIN = -2.0
LOGVAR_MAX = 2.0