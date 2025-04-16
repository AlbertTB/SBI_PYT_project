#!/usr/bin/env python3

#check if libraries are installed
try:
    import os
    import numpy as np
    import pandas as pd
    from Bio.PDB import PDBParser, DSSP
    from Bio.PDB.HSExposure import HSExposureCB
    from Bio.PDB.ResidueDepth import ResidueDepth
    from scipy.spatial import KDTree
    import multiprocessing as mp
    import warnings
    from tqdm import tqdm


except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please ensure all required libraries are installed.")
    raise


# Amino acid properties remain the same
# Hydrophobicity scale (Kyte & Doolittle)
hydrophobicity = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

# Size/volume of amino acids (in cubic Angstroms)
volume = {
    'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5,
    'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7,
    'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7,
    'SER': 89.0, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0
}

# Charge of amino acids at pH 7
charge = {
    'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
    'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0.1, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
}

# H-bond donor capacity
hbond_donor = {
    'ALA': 0, 'ARG': 5, 'ASN': 2, 'ASP': 1, 'CYS': 1,
    'GLN': 2, 'GLU': 1, 'GLY': 0, 'HIS': 2, 'ILE': 0,
    'LEU': 0, 'LYS': 2, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 1, 'THR': 1, 'TRP': 1, 'TYR': 1, 'VAL': 0
}

# H-bond acceptor capacity
hbond_acceptor = {
    'ALA': 0, 'ARG': 0, 'ASN': 2, 'ASP': 3, 'CYS': 0,
    'GLN': 2, 'GLU': 3, 'GLY': 0, 'HIS': 1, 'ILE': 0,
    'LEU': 0, 'LYS': 0, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 1, 'THR': 1, 'TRP': 0, 'TYR': 1, 'VAL': 0
}

def calculate_dssp(pdb_file, model):
    """Calculate DSSP with caching for better performance"""
            # Try using DSSP directly through Biopython
    dssp_data = DSSP(model, pdb_file)
    
    # Convert DSSP output to more convenient dict
    dssp = {}
    for key in dssp_data.keys():
        chain_id = key[0]
        res_id = key[1][1]  # Get residue number
        dssp[(chain_id, res_id)] = dssp_data[key]
    
    return dssp
    
def calculate_residue_depth(model):
    """Calculate residue depth using ResidueDepth while muting MSMS noise"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rd = ResidueDepth(model)
            return rd
    except Exception as e:
        print(f"Error calculating Residue Depth: {e}")
        return None

def extract_features(pdb_file, output_dir="features"):
    """
    Extract features from a PDB file for ligand binding site prediction - optimized version
    
    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file
    output_dir : str
        Directory to save the output features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the extracted features for each residue
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse PDB file - use custom parser options to improve speed
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    try:
        structure = parser.get_structure(pdb_id, pdb_file)
        model = structure[0]  # Get the first model
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return pd.DataFrame()
    
    # Get all atom coordinates for spatial calculations - do this only once
    all_atoms = []
    atom_residue_map = {}
    
    # Pre-collect all residues to avoid repeated iterations
    all_residues = []
    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            # Skip hetero and non-standard residues early
            res_id = residue.get_id()
            if res_id[0] != " " or residue.get_resname() not in hydrophobicity:
                continue
                
            all_residues.append((chain_id, residue))
    
    # Collect atom information in a single pass
    for chain_id, residue in all_residues:
        if residue.get_resname() == "HOH":  # Skip water molecules
            continue
        
        for atom in residue:
            atom_residue_map[atom.get_full_id()] = residue.get_id()
            all_atoms.append((atom.get_coord(), atom.get_full_id()))
    
    # Optimize neighbor search by using a more efficient KD-tree implementation
    coords = np.array([a[0] for a in all_atoms])
    atom_tree = KDTree(coords)
    
    # Calculate DSSP once for all residues
    dssp = calculate_dssp(pdb_file, model)
    
    # Calculate ResidueDepth efficiently
    
    rd = calculate_residue_depth(model)
    
    # Calculate HSE (half-sphere exposure)
    hse = None
    
    try:
        hse = HSExposureCB(model)
    except Exception:
        pass
    
    # Pre-calculate secondary structure types for one-hot encoding
    ss_types = ['H', 'B', 'E', 'G', 'I', 'T', 'S', ' ', 'X']
    
    # Process each residue efficiently
    features = []
    
    for chain_id, residue in all_residues:
        res_name = residue.get_resname()
        res_id = residue.get_id()[1]
        
        # Get residue coordinates (center of mass)
        residue_atoms = []
        ca_atom = None
        for atom in residue:
            if atom.get_id() == 'CA':  # Alpha carbon
                ca_atom = atom
            residue_atoms.append(atom.get_coord())
        
        if not residue_atoms or ca_atom is None:
            continue
            
        residue_center = np.mean(residue_atoms, axis=0)
        
        # 1. Basic residue properties - use direct dictionary access for speed
        res_features = {
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'residue_id': res_id,
            'residue_name': res_name,
            'hydrophobicity': hydrophobicity[res_name],
            'volume': volume[res_name],
            'charge': charge[res_name],
            'hbond_donor': hbond_donor[res_name],
            'hbond_acceptor': hbond_acceptor[res_name],
        }
        
        # 2. Structural features from DSSP
        dssp_key = (chain_id, res_id)
        if dssp and dssp_key in dssp:
            # DSSP provides: secondary structure, relative ASA, phi, psi angles, etc.
            dssp_data = dssp[dssp_key]
            
            # Corrected index: Secondary structure (index 2 instead of 1)
            res_features['ss_type'] = dssp_data[2]
            
            # Relative ASA (index 3)
            res_features['rel_asa'] = float(dssp_data[3])
            
            # Phi and Psi angles (indices 4 and 5)
            res_features['phi'] = float(dssp_data[4])
            res_features['psi'] = float(dssp_data[5])
        else:
            # Default values if DSSP calculation failed
            res_features['ss_type'] = 'X'
            res_features['rel_asa'] = 0.0
            res_features['phi'] = 0.0
            res_features['psi'] = 0.0
        
        # 3. HSE (Half-sphere exposure) - Corrected to use residue directly
        if hse:
            try:
                # Use the residue directly instead of key lookup
                hse_up = residue.xtra.get('EXP_HSE_B_U', None)
                hse_down = residue.xtra.get('EXP_HSE_B_D', None)

                res_features['hse_up'] = float(hse_up)
                res_features['hse_down'] = float(hse_down)
                res_features['hse_ratio'] = float(hse_up) / (float(hse_up) + float(hse_down)) if (float(hse_up) + float(hse_down)) > 0 else 0.0
            except:
                res_features['hse_up'] = 0.0
                res_features['hse_down'] = 0.0
                res_features['hse_ratio'] = 0.0
        else:
            res_features['hse_up'] = 0.0
            res_features['hse_down'] = 0.0
            res_features['hse_ratio'] = 0.0
        
        # 4. Residue depth
        if rd:
            resrd_id = residue.get_id()
            rd_key = (chain_id, resrd_id)
            try:
                rd_val, ca_depth = rd[rd_key]
                res_features['residue_depth'] = float(rd_val)
            except:
                res_features['residue_depth'] = 0.0
        else:
            res_features['residue_depth'] = 0.0

        
        # 5. & 6. Local environment features and residue protrusion
        # Optimize by combining multiple spatial queries into one
        ca_coords = ca_atom.get_coord()
        # Use a single query for both the 10Å and 8Å searches
        neighbors_10a = atom_tree.query_ball_point(residue_center, 10.0)
        # Filter the 10Å results to get 8Å results
        neighbors_8a = [i for i in neighbors_10a if np.linalg.norm(coords[i] - ca_coords) <= 8.0]
        
        # Process neighbor information
        neighbor_residues = set()
        pos_charged = 0
        neg_charged = 0
        
        for n_idx in neighbors_10a:
            full_id = all_atoms[n_idx][1]
            res_full_id = atom_residue_map[full_id]
            
            # Skip self
            if res_full_id[1] != res_id:
                neighbor_residues.add(res_full_id[1])
                
                # 7. Electrostatic features
                try:
                    # Try to safely get the residue
                    n_chain_id = full_id[2]
                    n_res = model[n_chain_id][res_full_id]
                    n_res_name = n_res.get_resname()
                    
                    if n_res_name in charge:
                        charge_val = charge[n_res_name]
                        if charge_val > 0:
                            pos_charged += 1
                        elif charge_val < 0:
                            neg_charged += 1
                except KeyError:
                    # Skip if we can't find this residue
                    continue
        
        res_features['neighbor_count'] = len(neighbor_residues)
        res_features['pos_charged_neighbors'] = pos_charged
        res_features['neg_charged_neighbors'] = neg_charged
        res_features['net_charge_environment'] = pos_charged - neg_charged
        
        # Calculate atom density from 8Å search
        atom_count = len(neighbors_8a)
        res_features['atom_density'] = atom_count / (4.0/3.0 * np.pi * 8.0**3)
        
        # 8. Convert secondary structure to numerical features using one-hot encoding
        # Optimize one-hot encoding
        ss = res_features['ss_type']
        for ss_type in ss_types:
            if ss_type == "-":
                res_features[f'ss_{ss_type}'] = 1
            res_features[f'ss_{ss_type}'] = 1 if ss == ss_type else 0
        
        # Add to features list
        features.append(res_features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    return df

def process_pdb_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    pdb_file, output_dir = args
    try:
        return extract_features(pdb_file, output_dir)
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return pd.DataFrame()

def process_directory(pdb_dir, output_dir="features"):
    """
    Process all PDB files in a directory using multiple processors
    
    Parameters:
    -----------
    pdb_dir : str
        Directory containing PDB files
    output_dir : str
        Directory to save feature files
    num_processes : int, optional
        Number of processes to use. Defaults to CPU count.
    
    Returns:
    --------
    list
        List of DataFrames with features for each PDB file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of PDB files
    pdb_files = [os.path.join(pdb_dir, file) for file in os.listdir(pdb_dir) if file.endswith(".pdb")]
    
    if not pdb_files:
        print(f"No PDB files found in {pdb_dir}")
        return []
    
    # Determine number of processes to use
    num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    
    # Process files in parallel
    print(f"Processing {len(pdb_files)} PDB files using {num_processes} processes...")
    
    args_list = [(pdb_file, output_dir) for pdb_file in pdb_files]
    
    all_features = []
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=len(args_list), desc="Extracting features", unit="file") as pbar:
            for features in pool.imap_unordered(process_pdb_file_wrapper, args_list):
                if not features.empty:
                    all_features.append(features)
                pbar.update(1)

    
    print(f"Processed {len(all_features)} PDB files successfully")
    return all_features

def label_binding_sites(features_df, binding_sites, output_file="labeled_features.csv"):
    """
    Label residues as binding or non-binding - optimized version
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing residue features
    binding_sites : dict
        Dictionary mapping PDB IDs to lists of binding residue IDs
        Format: {pdb_id: [(chain_id, res_id), ...], ...}
    output_file : str
        Path to save the labeled features
    
    Returns:
    --------
    pandas.DataFrame
        Labeled features DataFrame
    """
    if features_df.empty:
        print("No features to label")
        return features_df
    
    # Create a new column for binding site labels
    features_df['is_binding_site'] = 0
    
    # Create a more efficient lookup dictionary for binding sites
    binding_lookup = {}
    for pdb_id, sites in binding_sites.items():
        binding_lookup[pdb_id] = set((chain, res) for chain, res in sites)
    
    # Use vectorized operations where possible
    for pdb_id in features_df['pdb_id'].unique():
        if pdb_id in binding_lookup:
            # Create a mask for this PDB ID
            pdb_mask = features_df['pdb_id'] == pdb_id
            
            # Update binding sites for this PDB
            for i, row in features_df[pdb_mask].iterrows():
                chain_id = row['chain_id']
                res_id = int(row['residue_id'])
                
                if (chain_id, res_id) in binding_lookup[pdb_id]:
                    features_df.at[i, 'is_binding_site'] = 1
    
    # Save labeled features
    features_df.to_csv(output_file, index=False)
    
    # Print class distribution
    binding = features_df['is_binding_site'].sum()
    total = len(features_df)
    print(f"Binding sites: {binding} ({binding/total*100:.2f}%)")
    print(f"Non-binding sites: {total-binding} ({(total-binding)/total*100:.2f}%)")
    
    return features_df
