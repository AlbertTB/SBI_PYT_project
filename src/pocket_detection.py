#!/usr/bin/env python3
try:
    import os
    import numpy as np
    from scipy.spatial import Delaunay, ConvexHull, cKDTree
    from Bio.PDB import PDBParser, NeighborSearch
    from skimage import measure
    from sklearn.cluster import DBSCAN
    import pandas as pd
    from tqdm import tqdm
    import warnings
    
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please ensure all required libraries are installed.")
    raise
class GeometryBasedPocketFinder:
    """
    A class to identify potential binding pockets using geometric approaches.
    Implements alpha-shape detection, surface curvature analysis, and grid-based pocket detection.
    """
    
    def __init__(self, probe_radius=1.4, min_pocket_volume=100, grid_spacing=1.0, inclusion_radius=6.0):
        """
        Initialize the pocket finder with customizable parameters.
        
        Parameters:
        -----------
        probe_radius : float
            Radius of the probe sphere in Angstroms (water molecule is typically 1.4Å)
        min_pocket_volume : float
            Minimum volume of a valid pocket in cubic Angstroms
        grid_spacing : float
            Spacing of the 3D grid for grid-based pocket detection
        inclusion_radius : float
            Radius to include residues as part of a detected pocket
        """
        self.probe_radius = probe_radius
        self.min_pocket_volume = min_pocket_volume
        self.grid_spacing = grid_spacing
        self.inclusion_radius = inclusion_radius
        
    def analyze_protein(self, pdb_file, output_dir=None):
        """
        Analyze a protein structure to identify potential binding pockets.
        
        Parameters:
        -----------
        pdb_file : str
            Path to the PDB file
        output_dir : str, optional
            Directory to save output files
            
        Returns:
        --------
        dict
            Dictionary containing detected pockets and associated residues
        """
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Parse PDB file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        
        # Extract protein atoms (exclude water, ligands, etc.)
        protein_atoms = []
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Standard amino acid
                    for atom in residue:
                        if atom.element != 'H':  # Exclude hydrogens
                            protein_atoms.append(atom)
                            
        # Calculate surface using alpha shape (3D convex hull with constraints)
        surface_atoms, surface_vertices = self._identify_surface_atoms(protein_atoms)
        
        # Detect pockets using both alpha-shape and grid-based methods
        grid_pockets = self._detect_pockets_grid_based(protein_atoms)
        concavity_pockets = self._detect_pockets_by_concavity(surface_atoms, protein_atoms)
        
        # Consolidate pocket detections
        combined_pockets = self._consolidate_pockets(grid_pockets, concavity_pockets)
        
        # Calculate pocket properties
        pocket_properties = self._calculate_pocket_properties(combined_pockets, protein_atoms)
        
        # Identify residues associated with each pocket
        pocket_residues = self._identify_pocket_residues(model, pocket_properties)
        
        # Save visualizations if output directory is provided
        if output_dir:
            self._generate_visualizations(pdb_file, pocket_properties, output_dir)
        
        # Return results
        return {
            'pocket_properties': pocket_properties,
            'pocket_residues': pocket_residues
        }
    
    def _identify_surface_atoms(self, atoms):
        """
        Identify atoms on the protein surface using alpha shape detection.
        
        Returns:
        --------
        tuple
            Surface atoms and vertices
        """
        # Extract coordinates
        coords = np.array([atom.get_coord() for atom in atoms])
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(coords)
        except Exception as e:
            print(f"Error in Delaunay triangulation: {e}")
            # Fallback to simpler approach
            hull = ConvexHull(coords)
            surface_indices = list(set([i for s in hull.simplices for i in s]))
            return [atoms[i] for i in surface_indices], hull.simplices
        
        # Identify surface simplices (tetrahedra)
        surface_simplices = []
        for simplex in tri.simplices:
            # Calculate circumradius
            verts = coords[simplex]
            A = np.vstack((verts.T, np.ones(len(simplex))))
            b = np.sum(verts * verts, axis=1)
            try:
                weights = np.linalg.solve(A.T @ A, A.T @ b)
                center = weights[:-1]
                radius = np.sqrt(np.sum((verts[0] - center) ** 2))
            except np.linalg.LinAlgError:
                # If solve fails, skip this simplex (not useful for surface anyway)
                continue
            
            # Apply alpha criterion
            if radius > self.probe_radius * 2:  # Use probe_radius as threshold
                surface_simplices.append(simplex)
                
        # Extract unique surface atoms
        surface_indices = list(set([i for s in surface_simplices for i in s]))
        
        return [atoms[i] for i in surface_indices], surface_simplices
    
    def _detect_pockets_grid_based(self, atoms):
        """
        Detect pockets using a 3D grid approach.
        
        Returns:
        --------
        list
            List of potential pocket centers and their properties
        """
        # Create a 3D grid around the protein
        coords = np.array([atom.get_coord() for atom in atoms])
        
        # Define grid boundaries with padding
        padding = 5.0  # Angstroms of padding around protein
        min_coords = np.min(coords, axis=0) - padding
        max_coords = np.max(coords, axis=0) + padding
        
        # Create grid
        x = np.arange(min_coords[0], max_coords[0], self.grid_spacing)
        y = np.arange(min_coords[1], max_coords[1], self.grid_spacing)
        z = np.arange(min_coords[2], max_coords[2], self.grid_spacing)
        
        # Build KD-tree for efficient distance queries
        atom_tree = cKDTree(coords)
        
        # Create 3D occupancy grid
        grid = np.zeros((len(x), len(y), len(z)), dtype=bool)
        
        # Fill grid
        atom_radii = {
            'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85,
            'I': 1.98, 'H': 1.2
        }
        default_radius = 1.8  # Default for unknown atoms
        
        # Mark grid points that are inside the protein
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    point = np.array([xi, yj, zk])
                    
                    # Find nearest atom
                    distances, indices = atom_tree.query(point, k=1)
                    nearest_atom = atoms[indices]
                    
                    # Get atom radius
                    atom_radius = atom_radii.get(nearest_atom.element, default_radius)
                    
                    # Mark as occupied if inside atom + probe radius
                    if distances < (atom_radius + self.probe_radius):
                        grid[i, j, k] = True
        
        # Find cavities (non-occupied space surrounded by protein)
        # Label connected components of empty space
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labeled_grid, num_features = measure.label(~grid, return_num=True, connectivity=3)
        
        # Count boundary voxels
        boundary_voxels = (labeled_grid[:, :, 0] > 0).sum() + \
                          (labeled_grid[:, :, -1] > 0).sum() + \
                          (labeled_grid[:, 0, :] > 0).sum() + \
                          (labeled_grid[:, -1, :] > 0).sum() + \
                          (labeled_grid[0, :, :] > 0).sum() + \
                          (labeled_grid[-1, :, :] > 0).sum()
                          
        # The largest component is usually the outside
        if boundary_voxels > 0:
            # Find the component that touches the boundary
            boundary_labels = set()
            boundary_labels.update(labeled_grid[:, :, 0].flatten())
            boundary_labels.update(labeled_grid[:, :, -1].flatten())
            boundary_labels.update(labeled_grid[:, 0, :].flatten())
            boundary_labels.update(labeled_grid[:, -1, :].flatten())
            boundary_labels.update(labeled_grid[0, :, :].flatten())
            boundary_labels.update(labeled_grid[-1, :, :].flatten())
            
            # Remove 0 (background)
            if 0 in boundary_labels:
                boundary_labels.remove(0)
                
            # Remove outside component
            for label in boundary_labels:
                labeled_grid[labeled_grid == label] = 0
        else:
            # Just remove the largest component
            component_sizes = np.bincount(labeled_grid.flatten())
            largest_component = np.argmax(component_sizes[1:]) + 1
            labeled_grid[labeled_grid == largest_component] = 0
        
        # Extract pocket coordinates
        pockets = []
        for label in range(1, num_features + 1):
            if label in labeled_grid:
                # Get pocket voxels
                pocket_voxels = np.where(labeled_grid == label)
                
                # Convert to real coordinates
                pocket_coords = np.vstack([
                    x[pocket_voxels[0]],
                    y[pocket_voxels[1]],
                    z[pocket_voxels[2]]
                ]).T
                
                # Calculate pocket volume
                pocket_volume = len(pocket_voxels[0]) * self.grid_spacing**3
                
                # Only keep significant pockets
                if pocket_volume >= self.min_pocket_volume:
                    # Find pocket center
                    center = np.mean(pocket_coords, axis=0)
                    
                    pockets.append({
                        'center': center,
                        'coords': pocket_coords,
                        'volume': pocket_volume,
                        'method': 'grid-based'
                    })
        
        return pockets
    
    def _detect_pockets_by_concavity(self, surface_atoms, all_atoms):
        """
        Detect pockets by analyzing surface curvature and concavity.
        
        Returns:
        --------
        list
            List of potential pocket centers based on concavity
        """
        # Extract surface coordinates
        surface_coords = np.array([atom.get_coord() for atom in surface_atoms])
        
        if len(surface_coords) < 4:  # Need at least 4 points for 3D
            return []
            
        # Calculate local surface normals
        normals = []
        curvatures = []
        
        # Build KD-tree for neighborhood queries
        surface_tree = cKDTree(surface_coords)
        
        # Calculate surface normals and curvatures
        for i, point in enumerate(surface_coords):
            # Find nearest neighbors
            indices = surface_tree.query_ball_point(point, r=6.0)  # 6Å neighborhood
            
            if len(indices) < 4:
                # Not enough neighbors for reliable normal calculation
                normals.append(np.array([0, 0, 0]))
                curvatures.append(0)
                continue
                
            neighbors = surface_coords[indices]
            
            # Calculate center of neighborhood
            center = np.mean(neighbors, axis=0)
            
            # Calculate covariance matrix
            cov_mat = np.cov((neighbors - center).T)
            
            try:
                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
                
                # Normal is eigenvector with smallest eigenvalue
                normal = eigenvectors[:, 0]
                
                # Make normal point outward
                vec_to_center = point - center
                if np.dot(normal, vec_to_center) < 0:
                    normal = -normal
                    
                # Curvature metric (ratio of eigenvalues)
                curvature = eigenvalues[0] / (np.sum(eigenvalues) + 1e-10)
                
                normals.append(normal)
                curvatures.append(curvature)
            except np.linalg.LinAlgError:
                # Fallback for numerical instability
                normals.append(np.array([0, 0, 0]))
                curvatures.append(0)
        
        normals = np.array(normals)
        curvatures = np.array(curvatures)
        
        # Identify concave regions (potential binding sites)
        concave_indices = np.where(curvatures < np.percentile(curvatures, 15))[0]  # Bottom 15% curvature
        concave_points = surface_coords[concave_indices]
        
        if len(concave_points) < 5:
            return []
        
        # Cluster concave points to identify distinct pockets
        try:
            clustering = DBSCAN(eps=5.0, min_samples=3).fit(concave_points)
            
            labels = clustering.labels_
            unique_labels = set(labels)
            
            # Remove noise points (label -1)
            if -1 in unique_labels:
                unique_labels.remove(-1)
                
            pockets = []
            for label in unique_labels:
                cluster_points = concave_points[labels == label]
                
                # Calculate pocket center
                center = np.mean(cluster_points, axis=0)
                
                # Estimate volume based on convex hull
                if len(cluster_points) > 3:
                    try:
                        hull = ConvexHull(cluster_points)
                        volume = hull.volume
                    except:
                        # Fallback if convex hull fails
                        volume = len(cluster_points) * 8.0  # Rough estimate
                else:
                    volume = len(cluster_points) * 8.0
                
                pockets.append({
                    'center': center,
                    'coords': cluster_points,
                    'volume': volume,
                    'method': 'concavity'
                })
                
            return pockets
        
        except Exception as e:
            print(f"Error clustering concave points: {e}")
            return []
    
    def _consolidate_pockets(self, grid_pockets, concavity_pockets):
        """
        Consolidate pockets detected by different methods.
        
        Returns:
        --------
        list
            Consolidated list of unique pockets
        """
        # Combine all pockets
        all_pockets = grid_pockets + concavity_pockets
        
        # If no pockets were found, return empty list
        if not all_pockets:
            return []
            
        # Extract centers
        centers = np.array([pocket['center'] for pocket in all_pockets])
        
        # Cluster centers to identify unique pockets
        if len(centers) > 1:
            try:
                # Use clustering to group nearby pocket centers
                clustering = DBSCAN(eps=5.0, min_samples=1).fit(centers)
                labels = clustering.labels_
                
                unique_labels = set(labels)
                consolidated_pockets = []
                
                for label in unique_labels:
                    # Get all pockets in this cluster
                    cluster_indices = np.where(labels == label)[0]
                    cluster_pockets = [all_pockets[i] for i in cluster_indices]
                    
                    # Select the best pocket based on volume
                    best_pocket = max(cluster_pockets, key=lambda p: p['volume'])
                    
                    # Merge information from other pockets
                    best_pocket['evidence'] = len(cluster_indices)
                    
                    consolidated_pockets.append(best_pocket)
                    
                return consolidated_pockets
            except Exception as e:
                print(f"Error consolidating pockets: {e}")
                return all_pockets
        else:
            # Just one pocket, return as is
            return all_pockets
    
    def _calculate_pocket_properties(self, pockets, protein_atoms):
        """
        Calculate additional properties for each pocket.
        
        Returns:
        --------
        list
            Pockets with additional properties
        """
        # Extract protein atom coordinates
        protein_coords = np.array([atom.get_coord() for atom in protein_atoms])
        protein_tree = cKDTree(protein_coords)
        
        for pocket in pockets:
            center = pocket['center']
            
            # Calculate distance to nearest protein atom
            distances, indices = protein_tree.query(center, k=10)
            pocket['min_distance'] = np.min(distances)
            pocket['avg_distance'] = np.mean(distances)
            
            # Calculate hydrophobicity score
            hydrophobic_atoms = ['C']  # Carbon atoms are hydrophobic
            hydrophobic_count = sum(1 for i in indices if protein_atoms[i].element in hydrophobic_atoms)
            pocket['hydrophobicity_score'] = hydrophobic_count / 10.0  # Normalized
            
            # Calculate pocket depth (distance from surface to deepest point)
            # Use convex hull as a proxy for protein surface
            if len(protein_coords) > 3:
                try:
                    hull = ConvexHull(protein_coords)
                    hull_vertices = protein_coords[hull.vertices]
                    hull_tree = cKDTree(hull_vertices)
                    hull_distance, _ = hull_tree.query(center, k=1)
                    pocket['depth'] = hull_distance
                except:
                    pocket['depth'] = pocket['min_distance']
            else:
                pocket['depth'] = pocket['min_distance']
            
            # Calculate a score that combines volume, depth, and hydrophobicity
            pocket['score'] = (pocket['volume'] / 100.0) * pocket['depth'] * (0.5 + pocket['hydrophobicity_score'])
            
        # Sort pockets by score
        pockets.sort(key=lambda p: p['score'], reverse=True)
        
        return pockets
    
    def _identify_pocket_residues(self, model, pockets):
        """
        Identify residues that form each pocket.
        
        Returns:
        --------
        dict
            Dictionary mapping pocket indices to lists of residues
        """
        # Create a neighbor search for the entire model
        atoms = list(model.get_atoms())
        ns = NeighborSearch(atoms)
        
        pocket_residues = {}
        
        for i, pocket in enumerate(pockets):
            center = pocket['center']
            
            # Find residues within inclusion radius of the pocket center
            nearby_residues = set()
            center_neighboring_atoms = ns.search(center, self.inclusion_radius)
            
            for atom in center_neighboring_atoms:
                residue = atom.get_parent()
                chain_id = residue.get_parent().id
                res_id = residue.id[1]
                res_name = residue.resname
                nearby_residues.add((chain_id, res_id, res_name))
            
            # Store residues for this pocket
            pocket_residues[i] = list(nearby_residues)
        
        return pocket_residues
    
    def _generate_visualizations(self, pdb_file, pockets, output_dir):
        """
        Generate visualizations of detected pockets.
        
        Returns:
        --------
        None
            Saves visualization files to output directory
        """
        # Generate a simple PyMOL script to visualize pockets
        pdb_id = os.path.basename(pdb_file).split('.')[0]
        pymol_script = f"# PyMOL script for visualizing pockets in {pdb_id}\n"
        pymol_script += f"load {pdb_file}, protein\n"
        pymol_script += "show cartoon, protein\n"
        pymol_script += "color gray, protein\n"
        
        # Add spheres for pocket centers
        for i, pocket in enumerate(pockets):
            center = pocket['center']
            radius = (pocket['volume'] / (4.0 * np.pi / 3.0))**(1/3)  # Sphere radius based on volume
            pymol_script += f"pseudoatom pocket_{i}, pos=[{center[0]}, {center[1]}, {center[2]}]\n"
            pymol_script += f"show spheres, pocket_{i}\n"
            pymol_script += f"set sphere_scale, {min(radius/10, 3.0)}, pocket_{i}\n"
            color = f"0.{(i % 7) + 1}, 0.{(i % 5) + 1}, 0.{(i % 9) + 1}"  # Generate distinct colors
            pymol_script += f"color {color}, pocket_{i}\n"
        
        # Save PyMOL script
        pymol_file = os.path.join(output_dir, f"{pdb_id}_pockets.pml")
        with open(pymol_file, 'w') as f:
            f.write(pymol_script)
            
        # Save pocket information to CSV
        pocket_df = pd.DataFrame([
            {
                'pocket_id': i,
                'center_x': pocket['center'][0],
                'center_y': pocket['center'][1],
                'center_z': pocket['center'][2],
                'volume': pocket['volume'],
                'depth': pocket['depth'],
                'hydrophobicity': pocket['hydrophobicity_score'],
                'score': pocket['score'],
                'method': pocket['method']
            }
            for i, pocket in enumerate(pockets)
        ])
        
        pocket_df.to_csv(os.path.join(output_dir, f"{pdb_id}_pocket_info.csv"), index=False)
            
        print(f"Visualizations saved to {output_dir}")

def get_geometry_features(pdb_file, output_dir=None):
    """
    Extract geometry-based features for a protein structure.
    
    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    tuple
        (pocket_features_df, residue_features_df)
    """
    # Initialize pocket finder
    pocket_finder = GeometryBasedPocketFinder()
    
    # Analyze protein
    results = pocket_finder.analyze_protein(pdb_file, output_dir)
    
    # Extract pocket properties and residues
    pockets = results['pocket_properties']
    pocket_residues = results['pocket_residues']
    
    # Create a DataFrame with pocket features
    pocket_features = []
    for i, pocket in enumerate(pockets):
        pocket_features.append({
            'pocket_id': i,
            'volume': pocket['volume'],
            'depth': pocket['depth'],
            'hydrophobicity': pocket['hydrophobicity_score'],
            'min_distance': pocket['min_distance'],
            'avg_distance': pocket['avg_distance'],
            'score': pocket['score'],
            'method': pocket['method']
        })
        
    pocket_df = pd.DataFrame(pocket_features) if pocket_features else pd.DataFrame()
    
    # Create a DataFrame with residue features
    residue_features = []
    
    # Get PDB ID from filename
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    
    # Process each pocket's residues
    for pocket_id, residues in pocket_residues.items():
        pocket = pockets[pocket_id]
        
        for chain_id, res_id, res_name in residues:
            # Calculate distance to pocket center
            residue_features.append({
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'residue_id': res_id,
                'residue_name': res_name,
                'pocket_id': pocket_id,
                'pocket_volume': pocket['volume'],
                'pocket_depth': pocket['depth'],
                'pocket_hydrophobicity': pocket['hydrophobicity_score'],
                'pocket_score': pocket['score'],
                'is_in_pocket': 1  # This residue is in a pocket
            })
    
    # Create DataFrame
    residue_df = pd.DataFrame(residue_features) if residue_features else pd.DataFrame()
    
    return pocket_df, residue_df

def integrate_geometry_with_ml_features(ml_features_df, geometry_residue_df):
    """
    Combine machine learning features with geometry-based features.
    
    Parameters:
    -----------
    ml_features_df : pandas.DataFrame
        DataFrame containing ML-based features
    geometry_residue_df : pandas.DataFrame
        DataFrame containing geometry-based residue features
        
    Returns:
    --------
    pandas.DataFrame
        Combined features DataFrame
    """
    # Create a copy to avoid modifying the original
    combined_df = ml_features_df.copy()
    
    # Ensure residue_id is a string for matching
    combined_df['residue_id'] = combined_df['residue_id'].astype(int).astype(str)
    geometry_residue_df['residue_id'] = geometry_residue_df['residue_id'].astype(int).astype(str)

    # Default values for geometry features (for residues not in any pocket)
    geometry_defaults = {
        'pocket_id': -1,
        'pocket_volume': 0,
        'pocket_depth': 0,
        'pocket_hydrophobicity': 0,
        'pocket_score': 0,
        'is_in_pocket': 0
    }
    
    # Add geometry feature columns with default values
    for col, default_val in geometry_defaults.items():
        if col not in combined_df:
            combined_df[col] = default_val
    
    # If geometry features are empty, return defaults
    if geometry_residue_df.empty:
        return combined_df
    
    # Create a mapping key for both DataFrames
    combined_df['mapping_key'] = combined_df['pdb_id'] + '_' + combined_df['chain_id'] + '_' + combined_df['residue_id'].astype(str)
    geometry_residue_df['mapping_key'] = geometry_residue_df['pdb_id'] + '_' + geometry_residue_df['chain_id'] + '_' + geometry_residue_df['residue_id'].astype(str)
    
    # Create a dictionary for efficient lookups
    geometry_dict = {}
    for _, row in geometry_residue_df.iterrows():
        geometry_dict[row['mapping_key']] = {
            'pocket_id': row['pocket_id'],
            'pocket_volume': row['pocket_volume'],
            'pocket_depth': row['pocket_depth'],
            'pocket_hydrophobicity': row['pocket_hydrophobicity'],
            'pocket_score': row['pocket_score'],
            'is_in_pocket': row['is_in_pocket']
        }
    
    # Update combined DataFrame with geometry features
    for i, row in combined_df.iterrows():
        key = row['mapping_key']
        if key in geometry_dict:
            for col, val in geometry_dict[key].items():
                dtype = combined_df[col].dtype
                if np.issubdtype(dtype, np.integer):
                    val = int(val)
                elif np.issubdtype(dtype, np.floating):
                    val = float(val)
                combined_df.at[i, col] = val
    
    # Drop mapping key column
    combined_df.drop('mapping_key', axis=1, inplace=True)
    
    return combined_df

def calculate_pocket_properties(coords):
    """
    Calculate properties of a pocket from its coordinates.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Array of 3D coordinates forming the pocket
        
    Returns:
    --------
    dict
        Dictionary of pocket properties
    """
    if len(coords) < 4:
        return {
            'volume': 0,
            'surface_area': 0,
            'sphericity': 0
        }
    
    try:
        # Calculate convex hull
        hull = ConvexHull(coords)
        
        # Calculate volume
        volume = hull.volume
        
        # Calculate surface area
        surface_area = hull.area
        
        # Calculate sphericity (normalized ratio of volume to surface area)
        # Perfect sphere has sphericity 1, lower values indicate less spherical shapes
        sphere_volume = (4/3) * np.pi * ((3 * volume) / (4 * np.pi))**(1/3)
        sphere_area = 4 * np.pi * ((3 * volume) / (4 * np.pi))**(2/3)
        sphericity = (surface_area > 0) * (volume > 0) * ((6 * np.sqrt(np.pi) * volume**(2/3)) / surface_area)
        
        return {
            'volume': volume,
            'surface_area': surface_area,
            'sphericity': sphericity
        }
    except Exception as e:
        print(f"Error calculating pocket properties: {e}")
        return {
            'volume': 0,
            'surface_area': 0,
            'sphericity': 0
        }

def analyze_multiple_structures(pdb_dir, output_dir="geometry_features"):
    """
    Process a directory of PDB files to extract geometric features.
    
    Parameters:
    -----------
    pdb_dir : str
        Directory containing PDB files
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    pandas.DataFrame
        Combined residue features for all structures
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of PDB files
    pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    
    # Process each PDB file
    all_residue_features = []
    
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        try:
            # Get PDB ID from filename
            pdb_id = os.path.basename(pdb_file).split('.')[0]
            
            # Create subdirectory for this PDB
            pdb_output_dir = os.path.join(output_dir, pdb_id)
            os.makedirs(pdb_output_dir, exist_ok=True)
            
            # Extract geometry features
            _, residue_df = get_geometry_features(pdb_file, pdb_output_dir)
            
            if not residue_df.empty:
                all_residue_features.append(residue_df)
                
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            continue
    
    # Combine all residue features
    if all_residue_features:
        combined_df = pd.concat(all_residue_features, ignore_index=True)
        
        # Save combined features
        combined_output = os.path.join(output_dir, "all_residue_features.csv")
        combined_df.to_csv(combined_output, index=False)