import os
import pickle
import numpy as np
from tqdm import tqdm
import torc
import torch.nn.functional as F

Crystallization_aids = "SO4, GOL, EDO, PO4, ACT, PEG, DMS, TRS, PGE, PG4, FMT, EPE, MPD, MES, CD, IOD"
Ligand_exclusion = "144, 15P, 1PE, 2F2, 2JC, 3HR, 3SY, 7N5, 7PE, 9JE, AAE, ABA, ACE, ACN, ACT, ACY, AZI, BAM, BCN, BCT, BDN, BEN, BME, BO3, BTB, BTC, BU1, C8E, CAD, CAQ, CBM, CCN, CIT, CL, CLR, CM, CMO, CO3, CPT, CXS, D10, DEP, DIO, DMS, DN, DOD, DOX, EDO, EEE, EGL, EOH, EOX, EPE, ETF, FCY, FJO, FLC, FMT, FW5, GOL, GSH, GTT, GYF, HED, IHP, IHS, IMD, IOD, IPA, IPH, LDA, MB3, MEG, MES, MLA, MLI, MOH, MPD, MRD, MSE, MYR, N, NA, NH2, NH4, NHE, NO3, O4B, OHE, OLA, OLC, OMB, OME, OXA, P6G, PE3, PE4, PEG, PEO, PEP, PG0, PG4, PGE, PGR, PLM, PO4, POL, POP, PVO, SAR, SCN, SEO, SEP, SIN, SO4, SPD, SPM, SR, STE, STO, STU, TAR, TBU, TME, TPO, TRS, UNK, UNL, UNX, UPL, URE"
glycans = "045, 05L, 07E, 07Y, 08U, 09X, 0BD, 0H0, 0HX, 0LP, 0MK, 0NZ, 0UB, 0V4, 0WK, 0XY, 0YT, 10M, 12E, 145, 147, 149, 14T, 15L, 16F, 16G, 16O, 17T, 18D, 18O, 1CF, 1FT, 1GL, 1GN, 1LL, 1S3, 1S4, 1SD, 1X4, 20S, 20X, 22O, 22S, 23V, 24S, 25E, 26O, 27C, 289, 291, 293, 2DG, 2DR, 2F8, 2FG, 2FL, 2GL, 2GS, 2H5, 2HA, 2M4, 2M5, 2M8, 2OS, 2WP, 2WS, 32O, 34V, 38J, 3BU, 3DO, 3DY, 3FM, 3GR, 3HD, 3J3, 3J4, 3LJ, 3LR, 3MG, 3MK, 3R3, 3S6, 3SA, 3YW, 40J, 42D, 445, 44S, 46D, 46Z, 475, 48Z, 491, 49A, 49S, 49T, 49V, 4AM, 4CQ, 4GC, 4GL, 4GP, 4JA, 4N2, 4NN, 4QY, 4R1, 4RS, 4SG, 4UZ, 4V5, 50A, 51N, 56N, 57S, 5GF, 5GO, 5II, 5KQ, 5KS, 5KT, 5KV, 5L3, 5LS, 5LT, 5MM, 5N6, 5QP, 5SP, 5TH, 5TJ, 5TK, 5TM, 61J, 62I, 64K, 66O, 6BG, 6C2, 6DM, 6GB, 6GP, 6GR, 6K3, 6KH, 6KL, 6KS, 6KU, 6KW, 6LA, 6LS, 6LW, 6MJ, 6MN, 6PZ, 6S2, 6UD, 6YR, 6ZC, 73E, 79J, 7CV, 7D1, 7GP, 7JZ, 7K2, 7K3, 7NU, 83Y, 89Y, 8B7, 8B9, 8EX, 8GA, 8GG, 8GP, 8I4, 8LR, 8OQ, 8PK, 8S0, 8YV, 95Z, 96O, 98U, 9AM, 9C1, 9CD, 9GP, 9KJ, 9MR, 9OK, 9PG, 9QG, 9S7, 9SG, 9SJ, 9SM, 9SP, 9T1, 9T7, 9VP, 9WJ, 9WN, 9WZ, 9YW, A0K, A1Q, A2G, A5C, A6P, AAL, ABD, ABE, ABF, ABL, AC1, ACR, ACX, ADA, AF1, AFD, AFO, AFP, AGL, AH2, AH8, AHG, AHM, AHR, AIG, ALL, ALX, AMG, AMN, AMU, AMV, ANA, AOG, AQA, ARA, ARB, ARI, ARW, ASC, ASG, ASO, AXP, AXR, AY9, AZC, B0D, B16, B1H, B1N, B2G, B4G, B6D, B7G, B8D, B9D, BBK, BBV, BCD, BDF, BDG, BDP, BDR, BEM, BFN, BG6, BG8, BGC, BGL, BGN, BGP, BGS, BHG, BM3, BM7, BMA, BMX, BND, BNG, BNX, BO1, BOG, BQY, BS7, BTG, BTU, BW3, BWG, BXF, BXP, BXX, BXY, BZD, C3B, C3G, C3X, C4B, C4W, C5X, CBF, CBI, CBK, CDR, CE5, CE6, CE8, CEG, CEZ, CGF, CJB, CKB, CKP, CNP, CR1, CR6, CRA, CT3, CTO, CTR, CTT, D1M, D5E, D6G, DAF, DAG, DAN, DDA, DDL, DEG, DEL, DFR, DFX, DG0, DGO, DGS, DGU, DJB, DJE, DK4, DKX, DKZ, DL6, DLD, DLF, DLG, DNO, DO8, DOM, DPC, DQR, DR2, DR3, DR5, DRI, DSR, DT6, DVC, DYM, E3M, E5G, EAG, EBG, EBQ, EEN, EEQ, EGA, EMP, EMZ, EPG, EQP, EQV, ERE, ERI, ETT, EUS, F1P, F1X, F55, F58, F6P, F8X, FBP, FCA, FCB, FCT, FDP, FDQ, FFC, FFX, FIF, FK9, FKD, FMF, FMO, FNG, FNY, FRU, FSA, FSI, FSM, FSW, FUB, FUC, FUD, FUF, FUL, FUY, FVQ, FX1, FYJ, G0S, G16, G1P, G20, G28, G2F, G3F, G3I, G4D, G4S, G6D, G6P, G6S, G7P, G8Z, GAA, GAC, GAD, GAF, GAL, GAT, GBH, GC1, GC4, GC9, GCB, GCD, GCN, GCO, GCS, GCT, GCU, GCV, GCW, GDA, GDL, GE1, GE3, GFP, GIV, GL0, GL1, GL2, GL4, GL5, GL6, GL7, GL9, GLA, GLC, GLD, GLF, GLG, GLO, GLP, GLS, GLT, GM0, GMB, GMH, GMT, GMZ, GN1, GN4, GNS, GNX, GP0, GP1, GP4, GPH, GPK, GPM, GPO, GPQ, GPU, GPV, GPW, GQ1, GRF, GRX, GS1, GS9, GTK, GTM, GTR, GU0, GU1, GU2, GU3, GU4, GU5, GU6, GU8, GU9, GUF, GUL, GUP, GUZ, GXL, GXV, GYE, GYG, GYP, GYU, GYV, GZL, H1M, H1S, H2P, H3S, H53, H6Q, H6Z, HBZ, HD4, HNV, HNW, HSG, HSH, HSJ, HSQ, HSX, HSY, HTG, HTM, HVC, IAB, IDC, IDF, IDG, IDR, IDS, IDU, IDX, IDY, IEM, IN1, IPT, ISD, ISL, ISX, IXD, J5B, JFZ, JHM, JLT, JRV, JSV, JV4, JVA, JVS, JZR, K5B, K99, KBA, KBG, KD5, KDA, KDB, KDD, KDE, KDF, KDM, KDN, KDO, KDR, KFN, KG1, KGM, KHP, KME, KO1, KO2, KOT, KTU, L0W, L1L, L6S, L6T, LAG, LAH, LAI, LAK, LAO, LAT, LB2, LBS, LBT, LCN, LDY, LEC, LER, LFC, LFR, LGC, LGU, LKA, LKS, LM2, LMO, LNV, LOG, LOX, LRH, LTG, LVO, LVZ, LXB, LXC, LXZ, LZ0, M1F, M1P, M2F, M3M, M3N, M55, M6D, M6P, M7B, M7P, M8C, MA1, MA2, MA3, MA8, MAB, MAF, MAG, MAL, MAN, MAT, MAV, MAW, MBE, MBF, MBG, MCU, MDA, MDP, MFB, MFU, MG5, MGC, MGL, MGS, MJJ, MLB, MLR, MMA, MN0, MNA, MQG, MQT, MRH, MRP, MSX, MTT, MUB, MUR, MVP, MXY, MXZ, MYG, N1L, N3U, N9S, NA1, NAA, NAG, NBG, NBX, NBY, NDG, NFG, NG1, NG6, NGA, NGC, NGE, NGK, NGR, NGS, NGY, NGZ, NHF, NLC, NM6, NM9, NNG, NPF, NSQ, NT1, NTF, NTO, NTP, NXD, NYT, OAK, OI7, OPM, OSU, OTG, OTN, OTU, OX2, P53, P6P, P8E, PA1, PAV, PDX, PH5, PKM, PNA, PNG, PNJ, PNW, PPC, PRP, PSG, PSV, PTQ, PUF, PZU, QDK, QIF, QKH, QPS, QV4, R1P, R1X, R2B, R2G, RAE, RAF, RAM, RAO, RB5, RBL, RCD, RER, RF5, RG1, RGG, RHA, RHC, RI2, RIB, RIP, RM4, RP3, RP5, RP6, RR7, RRJ, RRY, RST, RTG, RTV, RUG, RUU, RV7, RVG, RVM, RWI, RY7, RZM, S7P, S81, SA0, SCG, SCR, SDY, SEJ, SF6, SF9, SFU, SG4, SG5, SG6, SG7, SGA, SGC, SGD, SGN, SHB, SHD, SHG, SIA, SID, SIO, SIZ, SLB, SLM, SLT, SMD, SN5, SNG, SOE, SOG, SOL, SOR, SR1, SSG, SSH, STW, STZ, SUC, SUP, SUS, SWE, SZZ, T68, T6D, T6P, T6T, TA6, TAG, TCB, TDG, TEU, TF0, TFU, TGA, TGK, TGR, TGY, TH1, TM5, TM6, TMR, TMX, TNX, TOA, TOC, TQY, TRE, TRV, TS8, TT7, TTV, TU4, TUG, TUJ, TUP, TUR, TVD, TVG, TVM, TVS, TVV, TVY, TW7, TWA, TWD, TWG, TWJ, TWY, TXB, TYV, U1Y, U2A, U2D, U63, U8V, U97, U9A, U9D, U9G, U9J, U9M, UAP, UBH, UBO, UDC, UEA, V3M, V3P, V71, VG1, VJ1, VJ4, VKN, VTB, W9T, WIA, WOO, WUN, WZ1, WZ2, X0X, X1P, X1X, X2F, X2Y, X34, X6X, X6Y, XDX, XGP, XIL, XKJ, XLF, XLS, XMM, XS2, XXM, XXR, XXX, XYF, XYL, XYP, XYS, XYT, XYZ, YDR, YIO, YJM, YKR, YO5, YX0, YX1, YYB, YYH, YYJ, YYK, YYM, YYQ, YZ0, Z0F, Z15, Z16, Z2D, Z2T, Z3K, Z3L, Z3Q, Z3U, Z4K, Z4R, Z4S, Z4U, Z4V, Z4W, Z4Y, Z57, Z5J, Z5L, Z61, Z6H, Z6J, Z6W, Z8H, Z8T, Z9D, Z9E, Z9H, Z9K, Z9L, Z9M, Z9N, Z9W, ZB0, ZB1, ZB2, ZB3, ZCD, ZCZ, ZD0, ZDC, ZDO, ZEE, ZEL, ZGE, ZMR"

VALID_RES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'SEC', 'UNK']
CRYSTALLIZATION_AIDS = {x.strip() for x in Crystallization_aids.split(",")}
LIGAND_EXCLUSION = {x.strip() for x in Ligand_exclusion.split(",")}
GLYCANS = {x.strip() for x in glycans.split(",")}
REMOVE_HET_NAMES = CRYSTALLIZATION_AIDS.union(LIGAND_EXCLUSION).union(GLYCANS)

RES_EMB = { 'ALA': 0,  'ARG': 1,  'ASN': 2,  'ASP': 3,  'CYS': 4, 'GLN': 5,  'GLU': 6,  'GLY': 7,  'HIS': 8,  'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'SEC': 20, 'UNK': 21 }

CHI_ANGLES = { 'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ'], ['CD', 'NE', 'CZ', 'NH1']], 'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']], 'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']], 'CYS': [['N', 'CA', 'CB', 'SG']], 'SEC': [['N', 'CA', 'CB', 'SE']], 'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']], 'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']], 'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']], 'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']], 'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']], 'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']], 'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']], 'SER': [['N', 'CA', 'CB', 'OG']], 'THR': [['N', 'CA', 'CB', 'OG1']], 'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'VAL': [['N', 'CA', 'CB', 'CG1']] }

MIN_NUM_PTS = 10

def parse_protein_structure(pdb_path):
    structure = {}
    with open(pdb_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                try:
                    chain = line[21]
                    res_seq = int(line[22:26].strip())
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    
                    if chain not in structure: structure[chain] = {}
                    if res_seq not in structure[chain]: structure[chain][res_seq] = {'res_name': res_name, 'atoms': {}}

                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    
                    structure[chain][res_seq]['atoms'][atom_name] = np.array([x, y, z])
                
                except (ValueError, IndexError):
                    # print(f"Skipping entire file due to malformed ATOM line: {pdb_path}") # 수정: 주석 처리
                    return None
    return structure

def get_atoms_from_structure(protein_structure, chain_id, resnum):
    try:
        residue_info = protein_structure[chain_id][resnum]
        return residue_info['res_name'], residue_info['atoms']
    except KeyError: return None, {}

def _calculate_dihedral_torch(p1, p2, p3, p4):
    """
    PyTorch를 사용하여 배치 단위로 이면각을 계산합니다.
    Args:
        p1, p2, p3, p4: 원자 좌표를 나타내는 (N, 3) 모양의 텐서.
    Returns:
        라디안 단위의 이면각을 가진 (N,) 모양의 텐서.
    """
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    
    n1 = F.normalize(n1, dim=-1, p=2)
    n2 = F.normalize(n2, dim=-1, p=2)
    b2 = F.normalize(b2, dim=-1, p=2)

    mask = torch.any(n1, dim=-1) & torch.any(n2, dim=-1)
    
    angle = torch.zeros(p1.shape[0], device=p1.device)

    if mask.any():
        x = torch.sum(n1[mask] * n2[mask], dim=-1)
        y = torch.sum(torch.cross(n1[mask], b2[mask], dim=-1) * n2[mask], dim=-1)
        x = torch.clamp(x, -1.0, 1.0)
        angle[mask] = torch.atan2(y, x)

    return angle

def get_angle_features(protein_structure, resdef):
    """
    주어진 포켓 내 잔기 리스트에 대해 백본(phi, psi, omega) 및 
    사이드체인(chi) 각도 특징을 배치 PyTorch 연산을 사용하여 계산합니다.
    
    Returns:
        각도의 sin/cos 값을 포함하는 (num_residues, 16) 모양의 numpy 배열.
    """
    num_res = len(resdef)
    if num_res == 0:
        return np.zeros((0, 16))

    device = torch.device("cpu")

    p1_phi, p2_phi, p3_phi, p4_phi = [torch.zeros(num_res, 3, device=device) for _ in range(4)]
    p1_psi, p2_psi, p3_psi, p4_psi = [torch.zeros(num_res, 3, device=device) for _ in range(4)]
    p1_omg, p2_omg, p3_omg, p4_omg = [torch.zeros(num_res, 3, device=device) for _ in range(4)]
    
    all_res_names = []
    
    def get_coord(chain, res_num, atom_name):
        try:
            return torch.tensor(protein_structure[chain][res_num]['atoms'][atom_name], device=device, dtype=torch.float32)
        except KeyError:
            return torch.zeros(3, device=device)

    for i, r_def in enumerate(resdef):
        chain_id, res_num = r_def[3:5].strip(), int(r_def[5:].strip())
        res_name, _ = get_atoms_from_structure(protein_structure, chain_id, res_num)
        all_res_names.append(res_name if res_name else 'UNK')

        c_prev = get_coord(chain_id, res_num - 1, 'C')
        n_curr = get_coord(chain_id, res_num, 'N')
        ca_curr = get_coord(chain_id, res_num, 'CA')
        c_curr = get_coord(chain_id, res_num, 'C')
        n_next = get_coord(chain_id, res_num + 1, 'N')
        ca_next = get_coord(chain_id, res_num + 1, 'CA')

        p1_phi[i], p2_phi[i], p3_phi[i], p4_phi[i] = c_prev, n_curr, ca_curr, c_curr
        p1_psi[i], p2_psi[i], p3_psi[i], p4_psi[i] = n_curr, ca_curr, c_curr, n_next
        p1_omg[i], p2_omg[i], p3_omg[i], p4_omg[i] = ca_curr, c_curr, n_next, ca_next

    phi = _calculate_dihedral_torch(p1_phi, p2_phi, p3_phi, p4_phi)
    psi = _calculate_dihedral_torch(p1_psi, p2_psi, p3_psi, p4_psi)
    omg = _calculate_dihedral_torch(p1_omg, p2_omg, p3_omg, p4_omg)

    pps_features = torch.stack([
        torch.sin(phi), torch.cos(phi),
        torch.sin(psi), torch.cos(psi),
        torch.sin(omg), torch.cos(omg)
    ], dim=1)

    chi_features = torch.zeros(num_res, 10, device=device)
    max_chi = max(len(v) for v in CHI_ANGLES.values()) if CHI_ANGLES else 0

    for j in range(max_chi):
        p1_chi, p2_chi, p3_chi, p4_chi = [torch.zeros(num_res, 3, device=device) for _ in range(4)]
        mask = torch.zeros(num_res, dtype=torch.bool, device=device)

        for i, r_def in enumerate(resdef):
            res_name = all_res_names[i]
            if res_name in CHI_ANGLES and j < len(CHI_ANGLES[res_name]):
                atom_names = CHI_ANGLES[res_name][j]
                chain_id, res_num = r_def[3:5].strip(), int(r_def[5:].strip())
                
                coords = [get_coord(chain_id, res_num, name) for name in atom_names]
                if not any(torch.all(c == 0) for c in coords):
                    p1_chi[i], p2_chi[i], p3_chi[i], p4_chi[i] = coords
                    mask[i] = True

        if mask.any():
            chi_j = _calculate_dihedral_torch(p1_chi[mask], p2_chi[mask], p3_chi[mask], p4_chi[mask])
            chi_features[mask, j * 2] = torch.sin(chi_j)
            chi_features[mask, j * 2 + 1] = torch.cos(chi_j)
    
    all_angle_features = torch.cat([chi_features, pps_features], dim=1)
    return all_angle_features.cpu().numpy()

def valid_ligand(pdb):
    with open(pdb, 'r') as p: lines = p.readlines()
    for line in lines:
        if line.startswith('ATOM'): break
        elif line.startswith('HETATM'): return line[17:20].strip() not in REMOVE_HET_NAMES
    return None

# 수정: 새로운 각도 계산 로직을 사용하도록 featurize 함수 변경
def featurize(pdb, invalid_site_pkl):
    if os.path.exists(invalid_site_pkl):
        with open(invalid_site_pkl, 'rb') as f:
            INVALID_SITE = pickle.load(f)
    else:
        # print(f"Warning: Invalid site pickle file not found at {invalid_site_pkl}. Assuming no invalid sites.") # 수정: 주석 처리
        INVALID_SITE = []
                
    linked_ligand = pdb.replace('_pocket', '')
    full_pdb = os.path.join(os.path.dirname(pdb), 'protein.pdb')
    if pdb in INVALID_SITE or not all(os.path.exists(p) for p in [linked_ligand, full_pdb]) or not valid_ligand(linked_ligand): return None
    
    protein_structure = parse_protein_structure(full_pdb)
    if protein_structure is None: return None
        
    with open(pdb, 'r') as p: resdef = sorted(list(set([line[17:26] for line in p.readlines()])))
    resdef = [r for r in resdef if len(r) >= 9 and r[0] != ' ']
    
    if not resdef: return None
        
    ca_coords, sc_coords_all = {}, {}
    valid_resdef = []
    for rdef in resdef:
        try:
            chain_id, res_num = rdef[3:5].strip(), int(rdef[5:].strip())
            res_info = protein_structure[chain_id][res_num]
            ca_coords[rdef] = res_info['atoms']['CA']
            sc_coords_all[rdef] = [coord for name, coord in res_info['atoms'].items() if name not in ['N', 'C', 'O', 'CA']]
            valid_resdef.append(rdef)
        except (KeyError, ValueError):
             continue
    
    resdef = valid_resdef
    if len(resdef) < MIN_NUM_PTS: return None

    sc_coords = {rdef: np.mean(crdList, axis=0) if crdList else ca_coords.get(rdef, np.zeros(3)) for rdef, crdList in sc_coords_all.items()}
    
    if len(ca_coords) != len(resdef): return None

    try:
        angle_features = get_angle_features(protein_structure, resdef)
    except Exception:
        return None

    coordinates = np.array([ca_coords[r] for r in resdef] + [sc_coords[r] for r in resdef])
    zero_center = coordinates - np.mean(coordinates, axis=0)
    max_dist = np.max(np.linalg.norm(zero_center, axis=1)) if len(zero_center) > 0 else 0.0
    scaling_factor = max_dist if max_dist > 1e-6 else 1.0
    scaled_coords = zero_center / scaling_factor
    
    ca_resCode = [RES_EMB.get(r[:3].strip(), RES_EMB['UNK']) for r in resdef]
    ca_resVec = np.zeros((len(ca_resCode), 22)); ca_resVec[np.arange(len(ca_resCode)), ca_resCode] = 1
    resVec = np.concatenate((ca_resVec, ca_resVec), axis=0)

    feat_scalar_list = []
    # CA atoms
    for i in range(len(resdef)):
        angles = angle_features[i]
        ftr = np.concatenate([resVec[i], angles, np.array([scaling_factor])])
        feat_scalar_list.append(ftr)

    # SC atoms
    for i in range(len(resdef)):
        angles = angle_features[i]
        ftr = np.concatenate([resVec[len(resdef) + i], angles, np.array([scaling_factor])])
        feat_scalar_list.append(ftr)
        
    feat_scalar = np.array(feat_scalar_list)
    return feat_scalar, scaled_coords