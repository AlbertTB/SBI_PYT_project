#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import glob
from collections import defaultdict
import joblib
import numpy as np
import tempfile
import shutil
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from pdb_feature_extraction import process_directory, label_binding_sites, extract_features
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pocket_detection import get_geometry_features, integrate_geometry_with_ml_features
from tqdm import tqdm

def enrich_features(pdb_file_and_df):
    pdb_file, df = pdb_file_and_df
    geometry_pocket_df, geometry_residue_df = get_geometry_features(pdb_file)
    return integrate_geometry_with_ml_features(df, geometry_residue_df)

def parse_prediction_file(file_path):
    """
    Parse a prediction file and extract binding site labels.
    
    Parameters:
    -----------
    file_path : str
        Path to the prediction file
        
    Returns:
    --------
    dict
        Dictionary mapping residue identifiers to binding class
    """
    df = pd.read_csv(file_path)
    
    # Create a dictionary to store binding site information
    binding_sites = {}
    
    for _, row in df.iterrows():
        # Parse the residue identifier (e.g., 1a0f_A_MET_1)
        res_id_parts = row['res_name'].split('_')
        
        # Format: pdb_id_chain_resname_resnum
        if len(res_id_parts) == 4:
            pdb_id = res_id_parts[0]
            chain_id = res_id_parts[1]
            res_num = int(res_id_parts[3])
            
            # Store the binding class
            binding_class = int(row['class'])
            
            # Create a unique identifier for this residue
            residue_key = (pdb_id, chain_id, res_num)
            binding_sites[residue_key] = binding_class
    
    return binding_sites

def process_all_prediction_files(prediction_dir):
    """
    Process all prediction files in a directory.
    
    Parameters:
    -----------
    prediction_dir : str
        Directory containing prediction files
        
    Returns:
    --------
    dict
        Dictionary mapping PDB IDs to binding site information
    """
    # Map of PDB IDs to binding site information
    all_binding_sites = defaultdict(dict)
    
    # Find all prediction files
    prediction_files = glob.glob(os.path.join(prediction_dir, "*.csv"))
    
    for file_path in prediction_files:
        binding_sites = parse_prediction_file(file_path)
        
        # Organize by PDB ID
        for (pdb_id, chain_id, res_num), binding_class in binding_sites.items():
            all_binding_sites[pdb_id][(chain_id, res_num)] = binding_class
    
    return all_binding_sites

def create_labeled_dataset(pdb_dir, prediction_dir, output_dir="labeled_features"):
    """
    Create a labeled dataset from PDB files and prediction files.

    Parameters:
    -----------
    pdb_dir : str
        Directory containing PDB files
    prediction_dir : str 
        Directory containing prediction files
        output_dir : str
        Directory to save labeled features

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing labeled features
    """

    # Number of PDB files in each batch
    batch_size = 50

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all prediction files to get binding sites
    binding_sites = process_all_prediction_files(prediction_dir)
    pdb_files = [
        os.path.join(pdb_dir, f"{pdb_id}.pdb")
        for pdb_id in binding_sites
        if os.path.exists(os.path.join(pdb_dir, f"{pdb_id}.pdb"))
    ]

    if not pdb_files:
        print("No matching PDB files found.")
        return None

    # Process PDB files in batches
    batch_csv_paths = []

    print(f"Processing {len(pdb_files)} PDB files in batches of {batch_size}...")

    num_processes = max(1, mp.cpu_count() - 1)

    for i in range(0, len(pdb_files), batch_size):
        batch_files = pdb_files[i:i + batch_size]
        batch_name = f"batch_{i // batch_size}"
        batch_csv = os.path.join(output_dir, f"{batch_name}.csv")

        with tempfile.TemporaryDirectory() as temp_dir:
            for pdb_file in batch_files:
                shutil.copy2(pdb_file, temp_dir)

            batch_dfs = process_directory(temp_dir, output_dir)

        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)

            # Add geometry features
            file_df_pairs = [
                (pdb_file, batch_df[batch_df['pdb_id'] == os.path.basename(pdb_file).split('.')[0]])
                for pdb_file in batch_files
            ]

            with mp.Pool(processes=num_processes) as pool:
                enriched_dfs = list(tqdm(pool.imap(enrich_features, file_df_pairs), total=len(file_df_pairs), desc="Enriching with geometry"))

            batch_df = pd.concat(enriched_dfs, ignore_index=True)
            batch_df.to_csv(batch_csv, index=False)
            batch_csv_paths.append(batch_csv)
            print(f"\u2714 Saved {len(batch_df)} residues to {batch_csv}")
        else:
            print(f"\u26a0 No features extracted in {batch_name}")

    if not batch_csv_paths:
        print("No batches were successfully processed.")
        return None

    print("\nCombining batch CSVs into one...")
    combined_df = pd.concat([pd.read_csv(f) for f in batch_csv_paths], ignore_index=True)
    combined_path = os.path.join(output_dir, "combined_features.csv")
    combined_df.to_csv(combined_path, index=False)

    # Label residues
    binding_sites_formatted = {
        pdb_id: [(chain_id, res_id) for (chain_id, res_id), binding_class in sites.items() if binding_class == 1]
        for pdb_id, sites in binding_sites.items()
    }

    labeled_path = os.path.join(output_dir, "combined_labeled_features.csv")
    labeled_df = label_binding_sites(combined_df, binding_sites_formatted, labeled_path)

    binding = labeled_df['is_binding_site'].sum()
    total = len(labeled_df)
    print(f"\nBinding sites: {binding} ({binding / total * 100:.2f}%)")
    print(f"Non-binding sites: {total - binding} ({(total - binding) / total * 100:.2f}%)")

    return labeled_df

def engineer_additional_features(df):
    """
    Create additional features that might help with binding site prediction
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the extracted features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional engineered features
    """
    # Create copy to avoid modifying original
    df_new = df.copy()   


    # Interaction terms between physicochemical properties
    df_new['hydro_volume'] = df_new['hydrophobicity'] * df_new['volume']
    df_new['charge_hbond'] = df_new['charge'] * (df_new['hbond_donor'] + df_new['hbond_acceptor'])
    
    # Surface accessibility features
    df_new['exposed_hydrophobic'] = df_new['hydrophobicity'] * df_new['rel_asa']
    df_new['exposed_charged'] = abs(df_new['charge']) * df_new['rel_asa']
    
    # Local environment complexity (entropy of neighbor properties)
    df_new['local_charge_imbalance'] = abs(df_new['pos_charged_neighbors'] - df_new['neg_charged_neighbors'])
    df_new['charge_density'] = df_new['net_charge_environment'] / (df_new['neighbor_count'] + 1)  # +1 to avoid division by zero
    
    # Structural pocket features
    df_new['inverse_depth'] = 1.0 / (df_new['residue_depth'] + 0.1)  # +0.1 to avoid division by zero
    df_new['depth_density'] = df_new['residue_depth'] * df_new['atom_density']
    
    # Residue conservation proxy (based on structural features)
    df_new['exposure_ratio'] = df_new['hse_up'] / (df_new['hse_down'] + 0.1)  # +0.1 to avoid division by zero
    
    # Hydrogen bonding potential in local environment
    df_new['hbond_potential'] = df_new['hbond_donor'] * df_new['hbond_acceptor']
    
    # Create aggregated secondary structure feature
    # Helix (H,G,I) - Sheet (E,B) - Loop (S,T, ,X)
    if 'ss_type' in df_new.columns:
        df_new['is_helix'] = df_new.apply(lambda x: 1 if x['ss_type'] in ['H', 'G', 'I'] else 0, axis=1)
        df_new['is_sheet'] = df_new.apply(lambda x: 1 if x['ss_type'] in ['E', 'B'] else 0, axis=1)
        df_new['is_loop'] = df_new.apply(lambda x: 1 if x['ss_type'] in ['S', 'T', ' ', 'X'] else 0, axis=1)
    
    # For columns that start with 'ss_', create interaction terms with other features
    ss_cols = [col for col in df_new.columns if col.startswith('ss_')]
    for ss_col in ss_cols:
        df_new[ss_col] = pd.to_numeric(df_new[ss_col], errors='coerce')
        df_new[f'{ss_col}_hydro'] = df_new[ss_col] * df_new['hydrophobicity']
        df_new[f'{ss_col}_asa'] = df_new[ss_col] * df_new['rel_asa']
    
    return df_new

def train_optimized_model(features_file, output_model="binding_site_model.pkl", cv_folds=5, perform_search=True):
    """
    Train an optimized model on the labeled dataset with SMOTE oversampling
    and hyperparameter tuning.
    
    Parameters:
    -----------
    features_file : str
        Path to the CSV file containing labeled features
    output_model : str
        Path to save the trained model
    cv_folds : int
        Number of cross-validation folds
    perform_search : bool
        Whether to perform hyperparameter search (slower but better results)
        
    Returns:
    --------
    tuple
        (trained_model, feature_importances, evaluation_metrics)
    """
    print(f"Training optimized model on {features_file}...")
    
    # Load the features
    df = pd.read_csv(features_file)
    
    # Add engineered features
    df = engineer_additional_features(df)
    
    groups = df['pdb_id']
    
    # Drop non-feature columns
    feature_columns = df.columns.tolist()
    non_feature_cols = ['pdb_id', 'chain_id', 'residue_id', 'residue_name', 'ss_type', 'is_binding_site']
    feature_columns = [col for col in df.columns if col not in non_feature_cols]
    
    # Check for columns with constant values and remove them
    constant_cols = []
    for col in feature_columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns: {', '.join(constant_cols)}")
        feature_columns = [col for col in feature_columns if col not in constant_cols]
    
    # Prepare features and target
    X = df[feature_columns].fillna(0)
    y = df['is_binding_site']
    
    # Create stratified group k-fold splits to account for protein-level grouping
    gkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Extract a single fold for final evaluation
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_groups = groups.iloc[test_idx]
        break
    
    # Build pipeline with preprocessing and SMOTE oversampling
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),  # Create synthetic examples for minority class
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=6))
    ])
    
    if perform_search:
        # Parameter grid for GridSearchCV
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 20, 30, 40],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],
            'smote__sampling_strategy': [0.3, 0.5, 0.7]  # Try different oversampling ratios
        }
        
        # Use inner cross-validation to tune hyperparameters
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        
        
        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, 
            n_iter=20,  # Much fewer combinations than full grid search
            cv=inner_cv, 
            scoring='recall', 
            verbose=1, 
            n_jobs=1,  # Reduce parallelism to save memory
            random_state=42
        )

        print("Performing randomized hyperparameter search...")
        random_search.fit(X_train, y_train, groups=groups.iloc[train_idx])
        best_pipeline = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")
    else:
        # Use default parameters with some improvements
        pipeline.set_params(
            classifier__n_estimators=300,
            classifier__max_depth=None,
            classifier__min_samples_split=5,
            classifier__min_samples_leaf=2,
            classifier__class_weight='balanced_subsample',
            classifier__max_features='sqrt',
            smote__sampling_strategy=0.5,
        )
        
        print("Fitting model with pre-defined parameters...")
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
    
    # Evaluate on test set
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics with focus on recall
    metrics = {
        'accuracy': np.mean(y_pred == y_test),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Get feature importances from the classifier
    rf_classifier = best_pipeline.named_steps['classifier']
    feature_importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_classifier.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 15 most important features:")
    print(feature_importances.head(15))
    
    # Save the model
    joblib.dump(best_pipeline, output_model, compress=3)
    print(f"\nModel saved to {output_model}")
    
    return best_pipeline, feature_importances, metrics

def run_prediction_pipeline(pdb_file, model_file="binding_site_model.pkl", output_dir="predictions"):
    """
    Run the prediction pipeline on a new PDB file and create a visualization PDB file.

    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file
    model_file : str
        Path to the trained model
    output_dir : str
        Directory to save the predictions and visualization PDB file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing predicted binding sites
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract features
    features_df = extract_features(pdb_file, output_dir)

    # Load the model
    rf = joblib.load(model_file)

    # Drop non-feature columns
    feature_columns = features_df.columns.tolist()
    non_feature_cols = ['pdb_id', 'chain_id', 'residue_id', 'residue_name', 'ss_type']

    for col in non_feature_cols:
        if col in feature_columns:
            feature_columns.remove(col)

    # Prepare features
    X = features_df[feature_columns].fillna(0)  # Replace NaN with 0

    # Make predictions
    predictions = rf.predict(X)
    probabilities = rf.predict_proba(X)[:, 1]

    # Add predictions to the DataFrame
    features_df['prediction'] = predictions
    features_df['probability'] = probabilities

    # Save predictions
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    output_file = os.path.join(output_dir, f"{pdb_id}_predictions.csv")

    # Create a more readable output format
    results_df = pd.DataFrame({
        'res_name': [f"{pdb_id}_{row['chain_id']}_{row['residue_name']}_{row['residue_id']}" for _, row in features_df.iterrows()],
        'prediction': features_df['prediction'],
        'probability': features_df['probability']
    })

    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Create a visualizable PDB file
    binding_sites = [
        (row['chain_id'], int(row['residue_id']))
        for _, row in features_df.iterrows() if row['prediction'] == 1
    ]
    output_pdb_file = os.path.join(output_dir, f"{pdb_id}_visualization.pdb")
    #create_visualization_pdb(pdb_file, binding_sites, output_pdb_file)

    return results_df
    #visualization in PyMOL: spectrum b, blue_white_red, minimum=0, maximum=1
    #Binding sites red, non-binding sites blue
    #In Chimera:
    #Go to Tools > Depiction > Render by Attribute.
    #Select B-factor as the attribute to render.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved binding site prediction pipeline")
    
    # Command group
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process predictions command
    process_parser = subparsers.add_parser("process", help="Process prediction files and extract features")
    process_parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB files")
    process_parser.add_argument("--prediction_dir", required=True, help="Directory containing prediction files")
    process_parser.add_argument("--output_dir", default="labeled_features", help="Directory to save labeled features")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train an optimized model")
    train_parser.add_argument("--features_file", required=True, help="CSV file with labeled features")
    train_parser.add_argument("--output_model", default="binding_site_model.pkl", help="Output model file")
    train_parser.add_argument("--random_search", action="store_true", help="Perform grid search for hyperparameters")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run predictions on a new PDB file")
    predict_parser.add_argument("--pdb_file", required=True, help="PDB file to predict on")
    predict_parser.add_argument("--model_file", default="binding_site_model.pkl", help="Trained model file")
    predict_parser.add_argument("--output_dir", default="predictions", help="Directory to save predictions")
    predict_parser.add_argument("--ensemble", action="store_true", help="Use ensemble prediction")
    
    args = parser.parse_args()
    
    if args.command == "process":
        create_labeled_dataset(args.pdb_dir, args.prediction_dir, args.output_dir)
    elif args.command == "train":
        train_optimized_model(args.features_file, args.output_model, perform_search=args.random_search)
    elif args.command == "predict":
        run_prediction_pipeline(args.pdb_file, args.model_file, args.output_dir)
    else:
        parser.print_help()