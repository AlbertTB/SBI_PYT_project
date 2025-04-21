# Tutorial for Predicting binding sites

## Requirements

### External programs

**IMPORTANT NOTE**: DSSP and MSMS need to be installed in order to get good prediction results.

DSSP:
```
sudo apt install dssp
mkdssp --version            #To test it
```

[MSMS](https://ccsb.scripps.edu/msms/downloads/): 

- Unpack it 
```
tar -xvzf msms.tar.gz
```

- Move binary to somewhere on your PATH 
```
sudo mv msms /usr/local/bin/
chmod +x /usr/local/bin/msms
```
**MAKE SURE THE NAME OF THE BINARY IS `msms`** (change name of binary if necessary)

### Libraries
- Core libraries:
```
numpy>=1.20.0

pandas>=1.3.0

scipy>=1.7.0
```
- Machine learning libraries:
```
scikit-learn>=0.24.0

imbalanced-learn>=0.8.0

joblib>=1.0.0
```
- Bioinformatics libraries:
```
biopython>=1.79

DSSP>=3.0.0  # If using the standalone DSSP binary
```
- Visualization libraries:

`scikit-image>=0.18.0`

- Progress tracking:

`tqdm>=4.61.0`

- Other utilities:

`multiprocessing>=0.70.12`

**To install this libraries use:**

``` pip install -r requirements.txt ```

### Additional requirements

Example of folder structure for ease of use (just for predicting binding sites, if you want to re-train the model see [Tutorial for training prediction model](#tutorial-for-training-prediction-model):

```
your_project/
├── predict_pdbs/          # PDB files that will be predicted
├── predictions_output/    # Auto-created: predictions for new inputs
├── models/                # Trained models
├── src/                   # Directory with the three main scripts (main, pocket_detection and feature_extraction)
```
## Predict binding sites
Run the following command (**Only one protein can be predicted at a time**):

```
python src/main.py predict --pdb_file predict_pdbs/new_protein.pdb --model_file models/name_of_model --output_dir predictions/ --log_file path/to/log_file.log (optional)
```
This will generate:

- A `.csv` file with prediction for all residues

- A `.csv` file with only the predicterd binding residues

- An annotated PDB file (optionally used for PyMOL or Chimera visualization)

## Test data

In the repository the folder named `test/` contains 4 proteins that can be used for testing the model. Their predictions are already located in the `predictions/` folder if you just want to visualise the results.

If you wish to test the model run the same command as before but instead of using the `predict_pdbs/` folder use the `test/` folder

# Tutorial for training prediction model

**Note**: Requirements are the same as in [Tutorial for Predicting Binding sites](#requirements) but the folder structure needs to be the following:

```
your_project/
├── predict_pdbs/          # PDB files that will be predicted
├── predictions_output/    # Auto-created: predictions for new inputs
├── models/                # Trained models
├── src/                   # Directory with the three main scripts (main, pocket_detection and feature_extraction)
├── train_data/            
    ├── protein/           # Contains the PDB files used to train the model
    ├── label/             # Contains CSVs with the class of each residue of each protein
├── labeled_features/      # Auto-created: CSVs with the features of the train_data
```

## First step: Feature extraction

This will extract geometric, structural adn physicochemical features for each protein and will label each residue.

Run the following command:
```
python src/main.py --process --pdb_dir train_data/protein --prediction_dir train_data/label --output_dir labeled_features/ (optional) --log_file path/to/log_file.log (optional)
```

This step may take a long time depending on the ammount of PDB files that need to be processed.

## Second step: Model training

To train a Random Forest model run the following command:
```
python src/main.py train --features_file labeled_features/combined_labeled_features.csv --output_model models/name_of_model (optional) --nested_cv (optional) --log_file path/to/log_file.log (optional)
```
Note that by default the model will be saved in the current directory

# Ligand Binding Site Prediction Program Documentation

This detailed report provides a comprehensive analysis of the ligand binding site prediction program, including its structure, algorithmic workflow, and the theoretical principles that underpin its functionality. The program combines geometric approaches with machine learning techniques to identify potential binding sites on protein structures with high accuracy.

## Program Structure and Organization

The program is organized into three primary Python modules, each with distinct responsibilities that collectively form a comprehensive binding site prediction pipeline:

### Core Modules

**pocket_detection.py**: Implements the `GeometryBasedPocketFinder` class that uses geometric algorithms to identify potential binding pockets in proteins. This module features multiple detection approaches:

- Alpha-shape detection for surface analysis
- 3D grid-based cavity identification
- Surface curvature and concavity analysis
- Pocket property calculation and scoring

**pdb_feature_extraction.py**: Focuses on extracting physicochemical and structural features from protein structures at the residue level, including:

- Amino acid properties (hydrophobicity, volume, charge)
- Secondary structure information via DSSP
- Solvent accessibility measurements
- Local environment characteristics
- Residue depth calculations

**main.py**: Serves as the orchestrator for the entire prediction pipeline, providing functionality for:

- Dataset preparation and feature enrichment
- Machine learning model training with cross-validation
- Prediction execution on new protein structures
- Visualization generation for identified binding sites


## Workflow of the Program

The binding site prediction process follows a systematic workflow that integrates geometric analysis with machine learning:

### 1. Protein Structure Parsing and Geometric Analysis

The process begins with parsing PDB files using BioPython's PDBParser. The program extracts the three-dimensional coordinates of atoms and performs geometric analyses to identify potential binding pockets using several complementary approaches:

```python
# Example from pocket_detection.py showing the analyze_protein method
def analyze_protein(self, pdb_file, output_dir=None):
    """
    Analyze a protein structure to identify potential binding pockets.
    """
    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    
    # Extract protein atoms
    protein_atoms = []
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] == ' ': # Standard amino acid
                for atom in residue:
                    if atom.element != 'H': # Exclude hydrogens
                        protein_atoms.append(atom)
    
    # Calculate surface using alpha shape
    surface_atoms, surface_vertices = self._identify_surface_atoms(protein_atoms)
    
    # Detect pockets using both alpha-shape and grid-based methods
    grid_pockets = self._detect_pockets_grid_based(protein_atoms)
    concavity_pockets = self._detect_pockets_by_concavity(surface_atoms, protein_atoms)
    
    # Consolidate pocket detections
    combined_pockets = self._consolidate_pockets(grid_pockets, concavity_pockets)
```


#### Alpha Shape Detection

This method delineates the protein surface by constructing Delaunay triangulation of atom coordinates and identifying surface simplices based on circumradius criteria[^1]. This approach is inspired by established pocket detection tools like Fpocket[^2].

#### Grid-Based Pocket Detection

The program creates a 3D grid surrounding the protein and marks grid points as occupied (inside protein) or unoccupied (potential pocket) based on their distance to the nearest atom. Connected components of unoccupied space surrounded by protein are identified as potential binding cavities[^3].

#### Surface Concavity Analysis

Surface atoms are analyzed for local curvature, and concave regions (low curvature values) are clustered using DBSCAN to identify distinct binding pockets [^4].

### 2. Feature Extraction

After identifying potential binding pockets, the program extracts a comprehensive set of features at the residue level:

```python
# Example from pdb_feature_extraction.py showing feature extraction
def extract_features(pdb_file, output_dir="features"):
    """
    Extract features from a PDB file for ligand binding site prediction
    """
    # Parse PDB file
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    try:
        structure = parser.get_structure(pdb_id, pdb_file)
        model = structure[0]  # Get the first model
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return pd.DataFrame()
    
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
    
    # Process each residue efficiently
    features = []
    for chain_id, residue in all_residues:
        # Extract various features...
        res_features = {
            'hydrophobicity': hydrophobicity[res_name],
            'volume': volume[res_name],
            'charge': charge[res_name],
            # ... many more features
        }
        features.append(res_features)
```

The program extracts the following key feature categories:

#### Physicochemical Properties

- Hydrophobicity using the Kyte \& Doolittle scale[^5]
- Residue volume in cubic Angstroms
- Charge states at physiological pH
- Hydrogen bond donor and acceptor capacities[^6]


#### Structural Features

- Secondary structure classification from DSSP[^7]
- Relative solvent accessibility
- Phi and Psi backbone angles
- Half-sphere exposure measurements
- Residue depth calculations[^8]


#### Local Environment Analysis

- Neighbor counts within defined radius
- Charged residue distribution
- Atom density
- Electrostatic environment[^6]


### 3. Feature Engineering and Model Training

The extracted features are enriched with additional derived features and used to train a machine learning model:

```python
# Example from main.py showing feature engineering
def engineer_additional_features(df):
    """
    Create additional features that might help with binding site prediction
    """
    # Create copy to avoid modifying original
    df_new = df.copy()
    
    # Interaction terms between physicochemical properties
    df_new['hydro_volume'] = df_new['hydrophobicity'] * df_new['volume']
    df_new['charge_hbond'] = df_new['charge'] * (df_new['hbond_donor'] + df_new['hbond_acceptor'])
    
    # Surface accessibility features
    df_new['exposed_hydrophobic'] = df_new['hydrophobicity'] * df_new['rel_asa']
    df_new['exposed_charged'] = abs(df_new['charge']) * df_new['rel_asa']
    
    # More engineered features...
    
    return df_new
```

The program employs a sophisticated machine learning pipeline with:

- StandardScaler for feature normalization
- SMOTE oversampling to address class imbalance
- RandomForestClassifier as the core prediction model
- Cross-validation using StratifiedGroupKFold to ensure robust evaluation[^9]


### 4. Prediction and Visualization

For new protein structures, the program applies the trained model to predict binding sites and generates visualization outputs:

```python
# Example from main.py showing prediction and visualization
def run_prediction_pipeline(pdb_file, model_file="binding_site_model.pkl", output_dir="predictions"):
    """
    Run the prediction pipeline on a new PDB file and create a visualization PDB file.
    """
    # Extract features
    features_df = extract_features(pdb_file, output_dir)
    geometry_pocket_df, geometry_residue_df = get_geometry_features(pdb_file)
    features_df = integrate_geometry_with_ml_features(features_df, geometry_residue_df)
    features_df = engineer_additional_features(features_df)
    
    # Load the model 
    rf = joblib.load(model_file)
    # Make predictions
    probabilities = rf.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.42).astype(int)
    
    # Create visualization file
    binding_sites = [(row['chain_id'], int(row['residue_id'])) 
                     for _, row in features_df.iterrows() if row['prediction'] == 1]
    create_visualization_pdb(pdb_file, binding_sites, output_pdb_file)
```

The program generates:

- Modified PDB files highlighting predicted binding sites
- PyMOL scripts for interactive visualization
- CSV files containing detailed prediction results


## Theoretical Foundations

### Structural Bioinformatics Principles

#### Alpha Shape Theory

Alpha shapes provide a formal mathematical framework for defining the shape of a set of points in 3D space. In the context of proteins, alpha shapes help identify the boundary between the protein interior and exterior, highlighting pockets and cavities.

The program implements alpha shape detection using the Delaunay triangulation of atom coordinates with filtering based on a probe radius (typically representing water molecules):

```python
def _identify_surface_atoms(self, atoms):
    """
    Identify atoms on the protein surface using alpha shape detection.
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
```

As noted by Schmidtke et al. in their work on Fpocket, this approach allows for efficient detection of potential binding sites on protein surfaces[^2].

#### Grid-Based Pocket Detection

The grid-based approach discretizes the 3D space around a protein into voxels and classifies them as inside/outside the protein based on distance thresholds. This method, similar to approaches used in computational geometry and computer graphics, enables the identification of enclosed cavities that might serve as binding sites[^3].

As demonstrated in the Pocket to Concavity (P2C) tool by Kudo et al., grid-based pocket detection is particularly effective for refining the shapes of predicted pockets to match the actual volume of bound ligands[^10].

#### Surface Curvature Analysis

Surface curvature analysis examines the local geometry of protein surfaces to identify concave regions that often correspond to binding sites. The program calculates local curvature using eigendecomposition of covariance matrices of surface neighborhoods[^4]:

```python
# Calculate covariance matrix
cov_mat = np.cov((neighbors - center).T)
# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
# Normal is eigenvector with smallest eigenvalue
normal = eigenvectors[:, 0]
# Curvature metric (ratio of eigenvalues)
curvature = eigenvalues[0] / (np.sum(eigenvalues) + 1e-10)
```

This approach is supported by research showing that binding sites typically exhibit high concavity compared to the rest of the protein surface[^2].

### Biochemical and Biophysical Foundations

#### Hydrophobicity and Binding Site Formation

Hydrophobic interactions are often the primary driving force in protein-ligand binding. The program incorporates the Kyte \& Doolittle hydrophobicity scale to characterize residues and evaluate their potential contribution to binding sites. Research by Elucidating the multiple roles of hydration for accurate protein-ligand binding prediction has demonstrated that the balance between hydrophobicity and hydration plays a crucial role in binding affinity[^5].

#### Solvent Accessibility and Binding Site Prediction

Solvent accessibility, calculated using DSSP, provides information about residue exposure to solvent. Binding sites typically show a distinctive pattern of solvent accessibility compared to the rest of the protein[^7]:

```python
# From pdb_feature_extraction.py
if dssp and dssp_key in dssp:
    # DSSP provides: secondary structure, relative ASA, phi, psi angles, etc.
    dssp_data = dssp[dssp_key]
    # Relative ASA (index 3)
    res_features['rel_asa'] = float(dssp_data)
```

The importance of solvent accessibility in binding site prediction is supported by numerous studies, including those on free energy calculations for protein-ligand binding prediction[^11].

#### Electrostatic Interactions and Complementarity

The program analyzes the electrostatic environment of residues by considering charge distributions in local neighborhoods:

```python
res_features['pos_charged_neighbors'] = pos_charged
res_features['neg_charged_neighbors'] = neg_charged
res_features['net_charge_environment'] = pos_charged - neg_charged
```

This analysis is crucial for detecting binding sites that involve charged or polar ligands, as demonstrated in studies on protein-ligand binding affinity prediction using deep learning models[^12].

### Machine Learning Principles

#### Random Forest for Binding Site Classification

The program employs Random Forest classification, an ensemble learning method that constructs multiple decision trees during training and outputs the mode of their individual predictions:

```python
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=random_state, sampling_strategy=0.5)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        max_features='sqrt',
        random_state=random_state,
        n_jobs=6
    ))
])
```

The effectiveness of Random Forest for binding site prediction is supported by multiple studies, including work by Deep Protein-Ligand Binding Prediction Using Unsupervised Learned Representations[^13].

#### Feature Importance Analysis

The program uses Random Forest's built-in feature importance analysis to identify the most predictive features for binding site classification:

```python
feature_importances = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)
```

This analysis helps understand the biochemical and structural determinants of ligand binding, aligning with findings from studies on protein-ligand binding affinity prediction via deep learning models.

#### Handling Class Imbalance with SMOTE

Binding site prediction typically involves highly imbalanced datasets, as binding residues constitute only a small fraction of the total residues. The program addresses this challenge using Synthetic Minority Over-sampling Technique (SMOTE):

```python
('smote', SMOTE(random_state=random_state, sampling_strategy=0.5))
```

This approach creates synthetic examples of the minority class (binding residues) to balance the dataset, improving model performance as demonstrated in recent machine learning approaches for binding site prediction[^14].

## Python Implementation Considerations

### Computational Efficiency

The program implements several optimizations to handle large protein structures efficiently:

1. **Optimized data structures**: KD-trees for spatial queries
```python
atom_tree = KDTree(coords)
```

2. **Parallel processing**: Multiprocessing for batch processing of PDB files
```python
with mp.Pool(processes=num_processes) as pool:
    enriched_dfs = list(tqdm(pool.imap(enrich_features, file_df_pairs), total=len(file_df_pairs), desc="Enriching with geometry"))
```

3. **Memory management**: Efficient handling of large datasets
```python
# Process PDB files in batches
for i in range(0, len(pdb_files), batch_size):
    batch_files = pdb_files[i:i + batch_size]
    batch_name = f"batch_{i // batch_size}"
```


### Modularity and Code Organization

The program follows object-oriented design principles, with clear separation of concerns:

1. The `GeometryBasedPocketFinder` class encapsulates geometric algorithms
2. Feature extraction is separated from machine learning in distinct modules
3. The main program provides a high-level interface through command-line arguments
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved binding site prediction pipeline")
    # Command group
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process predictions command
    process_parser = subparsers.add_parser("process", help="Process prediction files and extract features")
    process_parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB files")
```


## Conclusion

This ligand binding site prediction program represents a sophisticated integration of structural bioinformatics, biochemistry, and machine learning principles. By combining geometric analysis of protein structures with extensive feature extraction and advanced machine learning techniques, the program achieves accurate prediction of potential binding sites.

The modular design and comprehensive documentation make the program accessible for both research and practical applications in drug discovery and protein function analysis. The incorporation of established theoretical principles from structural bioinformatics and biochemistry, supported by relevant academic references, ensures the scientific validity of the prediction methodology.

## References

<div style="text-align: center">⁂</div>

[^1]: https://pubmed.ncbi.nlm.nih.gov/23193202/

[^2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2896101/

[^3]: https://pubmed.ncbi.nlm.nih.gov/37516103/

[^4]: https://pubmed.ncbi.nlm.nih.gov/30423105/

[^5]: https://pubmed.ncbi.nlm.nih.gov/10585500/

[^6]: https://www.biorxiv.org/content/10.1101/2024.10.07.616705v1

[^7]: https://www.biorxiv.org/content/10.1101/2025.04.11.648460v1

[^8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11641695/

[^9]: https://pubmed.ncbi.nlm.nih.gov/34741515/ 

[^10]: https://pubmed.ncbi.nlm.nih.gov/37086438/

[^11]: https://pubmed.ncbi.nlm.nih.gov/33206520/

[^12]: https://pubmed.ncbi.nlm.nih.gov/38529493/

[^13]: https://pubmed.ncbi.nlm.nih.gov/29028926/

[^14]: https://pubmed.ncbi.nlm.nih.gov/34322702/

















