import numpy as np

def parse_pdb(path):
    struct = {}
    try:
        with open(path, 'r') as f: lines = f.readlines()
    except Exception: return None
    for line in lines:
        if line.startswith("ATOM"):
            try:
                chain = line[21]; res_seq = int(line[22:26]); atom = line[12:16].strip()
                if chain not in struct: struct[chain] = {}
                if res_seq not in struct[chain]: struct[chain][res_seq] = {'atoms': {}}
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                struct[chain][res_seq]['atoms'][atom] = np.array([x, y, z])
            except (ValueError, IndexError): continue
    return struct