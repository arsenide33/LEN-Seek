# LEN-Seek: Accelerated Template-Based Protein-Ligand Binding Site Finder

LEN-Seek is a VAE-based Template-Based protein-ligand binding searching model.  
The model projects the structure data of ligand binding site into a point cloud on latent space.  
While evaluation, the model "compresses" the entire training data into 'latent' data, which is consisted of point clouds represnting each bidning site, and will serve as search space of similar binding sites. 
Abstract workflow is shown below:  
<img width="1899" height="540" alt="image" src="https://github.com/user-attachments/assets/8426abb8-b686-4b4b-ae66-747ebe12bcc3" />

Similarity of two datas is calculated with chamfer distance.  
Chamfer distance under 50.0 would be considered as similar binding site.
For tutorial, please check ```./example_run.ipynb```.

The model structure is same as below:
<img width="1689" height="573" alt="image" src="https://github.com/user-attachments/assets/9d888dd2-1a52-48c2-8667-c81c715109fb" />

## Presets
Dataset Format
To ensure the tool interprets the protein/binding site properly, the dataset MUST follow the directory structure below:

```
{DB directory}/
└── {pdbid}/
    ├── protein.pdb
    └── ligand_pocket_{pocketID}_{chainID}.pdb

# Example:
./YOUR_DB/
├── 1a2b/
│   ├── protein.pdb
│   └── ligand_pocket_1_A.pdb
└── 3xyz/
    ├── protein.pdb
    └── ligand_pocket_2_B.pdb
```

## Configuration 
You have to define some paths for database to process in config.py:
```
TRAIN_DB_PATH: Path to train database.

TRAIN_ANKH_PATH: Path to save Ankh embedding of train database.

EVAL_DB_PATH: Path to evaluation(query) database.

EVAL_ANKH_PATH: Path to save Ankh embedding of evalutaion(query) database.

CKPT_PATH: Path of model to be saved while training/loaded while evaluating. Leave this option unchanged unless if you want to train and utilize a new model. Modifications of paths below are not needed, unless you want to utilize precomputed data files.

TRAIN_DATA_PATH: Generated train data file from your database will be saved here. Change to './db/train_DB_precomputed.pkl' if you want to use precomputed one.

LATENT_DATA_PATH: Encoded train database by the model will be saved here. Change to './db/latent_DB_precomputed.pkl' if you want to use precomputed one.
Other options are recommended to be unchanged, unless you are training/evaluating a new model from the code.
```
* precomputed data files are currently not uploaded due to file size limit. If you want the precomputed data file, please contact here: arsenide33@snu.ac.kr
