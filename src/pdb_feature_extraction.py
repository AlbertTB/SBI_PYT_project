#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.HSExposure import HSExposureCB
from Bio.PDB.ResidueDepth import ResidueDepth
import warnings
from scipy.spatial import KDTree
from collections import defaultdict


# Amino acid properties
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

# In pdb_feature_extraction.py, modify the atom_residue_map to store full residue IDs:

def extract_features(pdb_file, output_dir="features"):
    """
    Extract features from a PDB file for ligand binding site prediction
    
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parse PDB file
    parser = PDBParser()
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]  # Get the first model
    
    # Calculate residue features
    features = []
    
    # Get all atom coordinates for spatial calculations
    all_atoms = []
    atom_residue_map = {}
    
    for residue in model.get_residues():
        if residue.get_resname() == "HOH":  # Skip water molecules
            continue
        
        for atom in residue:
            # Store the full residue ID to properly map back later
            atom_residue_map[atom.get_full_id()] = residue.get_id()
            all_atoms.append((atom.get_coord(), atom.get_full_id()))
    
    # Create KD-tree for efficient neighbor search
    coords = np.array([a[0] for a in all_atoms])
    atom_tree = KDTree(coords)
    
    # Skip DSSP and ResidueDepth since they're causing issues
    dssp = None
    rd = None
    
    # Calculate HSE (half-sphere exposure)
    try:
        hse = HSExposureCB(model)
    except Exception as e:
        print(f"HSE calculation failed for {pdb_file}: {e}")
        hse = None
    
    # Process each residue
    for chain in model:
        chain_id = chain.get_id()
        
        for residue in chain:
            if residue.get_id()[0] != " ":  # Skip hetero-residues
                continue
            
            res_name = residue.get_resname()
            res_id = residue.get_id()[1]
            
            if res_name not in hydrophobicity:  # Skip non-standard amino acids
                continue
            
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
            
            # 1. Basic residue properties
            res_features = {
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'residue_id': res_id,
                'residue_name': res_name,
                'hydrophobicity': hydrophobicity.get(res_name, 0),
                'volume': volume.get(res_name, 0),
                'charge': charge.get(res_name, 0),
                'hbond_donor': hbond_donor.get(res_name, 0),
                'hbond_acceptor': hbond_acceptor.get(res_name, 0),
            }
            
            # 2. Structural features (set defaults since DSSP might be disabled)
            res_features['ss_type'] = 'X'
            res_features['rel_asa'] = 0.0
            res_features['phi'] = 0.0
            res_features['psi'] = 0.0
            
            # 3. HSE (Half-sphere exposure)
            if hse:
                hse_key = (chain_id, res_id)
                try:
                    hse_up, hse_down = hse[hse_key][0]
                    res_features['hse_up'] = float(hse_up)
                    res_features['hse_down'] = float(hse_down)
                    res_features['hse_ratio'] = float(hse_up) / (float(hse_down) + 0.1)  # +0.1 to avoid division by zero
                except:
                    res_features['hse_up'] = 0.0
                    res_features['hse_down'] = 0.0
                    res_features['hse_ratio'] = 0.0
            else:
                res_features['hse_up'] = 0.0
                res_features['hse_down'] = 0.0
                res_features['hse_ratio'] = 0.0
            
            # 4. Residue depth (defaults since we're skipping rd calculation)
            res_features['residue_depth'] = 0.0
            res_features['ca_depth'] = 0.0
            
            # 5. Local environment features
            # Find neighboring atoms within 10 Angstroms
            neighbors = atom_tree.query_ball_point(residue_center, 10.0)
            neighbor_residues = set()
            
            for n_idx in neighbors:
                full_id = all_atoms[n_idx][1]
                res_full_id = atom_residue_map[full_id]
                # Use the residue number for comparison
                if res_full_id[1] != res_id:  # Skip self
                    neighbor_residues.add(res_full_id[1])
            
            res_features['neighbor_count'] = len(neighbor_residues)
            
            # 6. Calculate residue protrusion
            ca_coords = ca_atom.get_coord()
            neighbors_8a = atom_tree.query_ball_point(ca_coords, 8.0)
            
            # Count atoms in local environment
            atom_count = len(neighbors_8a)
            res_features['atom_density'] = atom_count / (4.0/3.0 * np.pi * 8.0**3)
            
            # 7. Electrostatic features
            pos_charged = 0
            neg_charged = 0
            
            for n_idx in neighbors:
                n_atom_full_id = all_atoms[n_idx][1]
                n_res_full_id = atom_residue_map[n_atom_full_id]
                
                # Skip self - compare residue numbers
                if n_res_full_id[1] == res_id:
                    continue
                
                try:
                    # Try to safely get the residue
                    n_chain_id = n_atom_full_id[2]
                    n_res = model[n_chain_id][n_res_full_id]
                    n_res_name = n_res.get_resname()
                    
                    if n_res_name in charge:
                        if charge[n_res_name] > 0:
                            pos_charged += 1
                        elif charge[n_res_name] < 0:
                            neg_charged += 1
                except KeyError:
                    # Skip if we can't find this residue
                    continue
            
            res_features['pos_charged_neighbors'] = pos_charged
            res_features['neg_charged_neighbors'] = neg_charged
            res_features['net_charge_environment'] = pos_charged - neg_charged
            
            # 8. Convert secondary structure to numerical features using one-hot encoding
            ss_types = ['H', 'B', 'E', 'G', 'I', 'T', 'S', ' ', 'X']
            for ss in ss_types:
                res_features[f'ss_{ss}'] = 1 if res_features['ss_type'] == ss else 0
            
            # Add to features list
            features.append(res_features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    
    # Save features to CSV
    output_file = os.path.join(output_dir, f"{pdb_id}_features.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Extracted {len(df)} residue features from {pdb_id}")
    return df

def process_directory(pdb_dir, output_dir="features"):
    """
    Process all PDB files in a directory
    
    Parameters:
    -----------
    pdb_dir : str
        Directory containing PDB files
    output_dir : str
        Directory to save feature files
    
    Returns:
    --------
    list
        List of DataFrames with features for each PDB file
    """
    all_features = []
    
    for file in os.listdir(pdb_dir):
        if file.endswith(".pdb"):
            pdb_file = os.path.join(pdb_dir, file)
            try:
                features = extract_features(pdb_file, output_dir)
                all_features.append(features)
                print(f"Processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    return all_features

def combine_features(feature_dfs, output_file="combined_features.csv"):
    """
    Combine all feature DataFrames into a single DataFrame
    
    Parameters:
    -----------
    feature_dfs : list
        List of feature DataFrames
    output_file : str
        Path to save the combined features
    
    Returns:
    --------
    pandas.DataFrame
        Combined features DataFrame
    """
    combined = pd.concat(feature_dfs, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"Combined features saved to {output_file}")
    return combined

def label_binding_sites(features_df, binding_sites, output_file="labeled_features.csv"):
    """
    Label residues as binding or non-binding
    
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
    # Create a new column for binding site labels
    features_df['is_binding_site'] = 0
    
    # Label binding sites
    for i, row in features_df.iterrows():
        pdb_id = row['pdb_id']
        chain_id = row['chain_id']
        res_id = int(row['residue_id'])
        
        if pdb_id in binding_sites:
            if (chain_id, res_id) in binding_sites[pdb_id]:
                features_df.at[i, 'is_binding_site'] = 1
    
    # Save labeled features
    features_df.to_csv(output_file, index=False)
    print(f"Labeled features saved to {output_file}")
    
    # Print class distribution
    binding = features_df['is_binding_site'].sum()
    total = len(features_df)
    print(f"Binding sites: {binding} ({binding/total*100:.2f}%)")
    print(f"Non-binding sites: {total-binding} ({(total-binding)/total*100:.2f}%)")
    
    return features_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from PDB files for ligand binding site prediction')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Directory containing PDB files')
    parser.add_argument('--output_dir', type=str, default='features', help='Directory to save feature files')
    parser.add_argument('--binding_sites', type=str, help='Path to binding sites file (optional)')
    args = parser.parse_args()
    
    # Process PDB files
    feature_dfs = process_directory(args.pdb_dir, args.output_dir)
    
    # Combine features
    combined_features = combine_features(feature_dfs, os.path.join(args.output_dir, "combined_features.csv"))
    
    # Label binding sites if provided
    if args.binding_sites:
        import json
        with open(args.binding_sites, 'r') as f:
            binding_sites = json.load(f)
        
        labeled_features = label_binding_sites(
            combined_features, 
            binding_sites, 
            os.path.join(args.output_dir, "labeled_features.csv")
        )
        
        print("Features extracted and labeled successfully!")
    else:
        print("Features extracted successfully!")