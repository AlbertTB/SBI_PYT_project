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
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from pdb_feature_extraction import process_directory, label_binding_sites, extract_features
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pocket_detection import get_geometry_features, integrate_geometry_with_ml_features
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def enrich_features(pdb_file_and_df):
    """
    Enrich features with geometry information from the PDB file.

    Parameters:
    -----------
    pdb_file_and_df : tuple
        Tuple containing the PDB file path and the DataFrame with features

    Returns:
    --------
    pandas.DataFrame
        DataFrame with enriched features
    """

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
        logger.warning("No matching PDB files found.")
        return None

    # Process PDB files in batches
    batch_csv_paths = []

    logger.info(f"Processing {len(pdb_files)} PDB files in batches of {batch_size}...")

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
            logger.info(f"✓ Saved {len(batch_df)} residues to {batch_csv}")
        else:
            logger.warning(f"⚠ No features extracted in {batch_name}")

    if not batch_csv_paths:
        logger.error("No batches were successfully processed.")
        return None

    logger.info("\nCombining batch CSVs into one...")
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
    logger.info(f"\nBinding sites: {binding} ({binding / total * 100:.2f}%)")
    logger.info(f"Non-binding sites: {total - binding} ({(total - binding) / total * 100:.2f}%)")

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

def train_optimized_model(features_file, output_model="binding_site_model.pkl", cv_folds=5, 
                          nested_cv=True, test_size=0.2, random_state=42):
    """
    Train an optimized model on the labeled dataset with SMOTE oversampling
    and proper cross-validation.
    
    Parameters:
    -----------
    features_file : str
        Path to the CSV file containing labeled features
    output_model : str
        Path to save the trained model
    cv_folds : int
        Number of cross-validation folds
    nested_cv : bool
        Whether to use nested cross-validation with a separate test set
    test_size : float
        Proportion of the dataset to include in the test split if nested_cv is True
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (trained_model, feature_importances, evaluation_metrics)
    """
    logger.info(f"Training optimized model on {features_file}...")
    
    # Load the features
    df = pd.read_csv(features_file)
    
    # Add engineered features
    df = engineer_additional_features(df)
    
    # Get groups for stratification
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
        logger.info(f"Removing {len(constant_cols)} constant columns: {', '.join(constant_cols)}")
        feature_columns = [col for col in feature_columns if col not in constant_cols]
    
    # Prepare features and target
    X = df[feature_columns].fillna(0)
    y = df['is_binding_site']
    
    # Create pipeline with preprocessing and SMOTE oversampling
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
    
    if nested_cv:
        # Implement nested cross-validation
        # Outer loop: final evaluation
        # Inner loop: model selection/hyperparameter tuning
        
        # First split data into training and final test set
        # Use GroupShuffleSplit to ensure that proteins don't overlap between train and test
        
        
        outer_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(outer_split.split(X, y, groups=groups))
        
        X_train_outer, X_test_final = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_final = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        
        logger.info(f"Split data into training set ({len(X_train_outer)} samples) and final test set ({len(X_test_final)} samples)")
        logger.info(f"Training set has {y_train_outer.sum()} binding sites ({y_train_outer.sum()/len(y_train_outer)*100:.2f}%)")
        logger.info(f"Test set has {y_test_final.sum()} binding sites ({y_test_final.sum()/len(y_test_final)*100:.2f}%)")
        
        # Inner cross-validation for model selection
        inner_cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Collect metrics from each fold
        fold_metrics = []
        logger.info(f"\nPerforming {cv_folds}-fold cross-validation on training set...")
        
        for fold, (train_idx, val_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer, groups_train)):
            X_train, X_val = X_train_outer.iloc[train_idx], X_train_outer.iloc[val_idx]
            y_train, y_val = y_train_outer.iloc[train_idx], y_train_outer.iloc[val_idx]
            
            logger.info(f"\nFold {fold+1}/{cv_folds}:")
            logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = pipeline.predict(X_val)
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            fold_metric = {
                'fold': fold+1,
                'accuracy': np.mean(y_pred == y_val),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_prob)
            }
            
            fold_metrics.append(fold_metric)
            logger.info(f"Fold {fold+1} metrics: Precision={fold_metric['precision']:.4f}, Recall={fold_metric['recall']:.4f}, F1={fold_metric['f1']:.4f}")
        
        # Average cross-validation metrics
        avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) 
                      for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
        
        logger.info("\nCross-validation results:")
        for metric, value in avg_metrics.items():
            logger.info(f"Average {metric}: {value:.4f}")
        
        # Now train the final model on the entire training set
        logger.info("\nTraining final model on entire training set...")
        pipeline.fit(X_train_outer, y_train_outer)
        
        # Evaluate on the held-out test set
        y_pred_final = pipeline.predict(X_test_final)
        y_prob_final = pipeline.predict_proba(X_test_final)[:, 1]
        
        final_metrics = {
            'accuracy': np.mean(y_pred_final == y_test_final),
            'precision': precision_score(y_test_final, y_pred_final),
            'recall': recall_score(y_test_final, y_pred_final),
            'f1': f1_score(y_test_final, y_pred_final),
            'roc_auc': roc_auc_score(y_test_final, y_prob_final)
        }
        
        logger.info("\nFinal Test Set Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nClassification Report on Test Set:")
        logger.info("\n" + classification_report(y_test_final, y_pred_final))
        
        logger.info("\nConfusion Matrix on Test Set:")
        logger.info("\n" + str(confusion_matrix(y_test_final, y_pred_final)))
    
    else:
        # Standard k-fold cross-validation
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Collect metrics from each fold
        fold_metrics = []
        all_true = []
        all_pred = []
        all_prob = []
        
        logger.info(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            logger.info(f"\nFold {fold+1}/{cv_folds}:")
            logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Store predictions for overall metrics
            all_true.extend(y_test)
            all_pred.extend(y_pred)
            all_prob.extend(y_prob)
            
            # Calculate metrics
            fold_metric = {
                'fold': fold+1,
                'accuracy': np.mean(y_pred == y_test),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            fold_metrics.append(fold_metric)
            logger.info(f"Fold {fold+1} metrics: Precision={fold_metric['precision']:.4f}, Recall={fold_metric['recall']:.4f}, F1={fold_metric['f1']:.4f}")
        
        # Average cross-validation metrics
        avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) 
                      for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
        
        logger.info("\nCross-validation results:")
        for metric, value in avg_metrics.items():
            logger.info(f"Average {metric}: {value:.4f}")
        
        # Overall metrics across all folds
        overall_metrics = {
            'accuracy': np.mean(np.array(all_pred) == np.array(all_true)),
            'precision': precision_score(all_true, all_pred),
            'recall': recall_score(all_true, all_pred),
            'f1': f1_score(all_true, all_pred),
            'roc_auc': roc_auc_score(all_true, all_prob)
        }
        
        logger.info("\nOverall metrics across all folds:")
        for metric, value in overall_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nOverall Classification Report:")
        logger.info("\n" + classification_report(all_true, all_pred))
        
        logger.info("\nOverall Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(all_true, all_pred)))
        
        # Train the final model on the entire dataset
        logger.info("\nTraining final model on entire dataset...")
        pipeline.fit(X, y)
        final_metrics = avg_metrics
    
    # Get feature importances from the classifier
    rf_classifier = pipeline.named_steps['classifier']
    feature_importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_classifier.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    logger.info("\nTop 15 most important features:")
    logger.info("\n" + str(feature_importances.head(15)))
    
    # Save the model
    joblib.dump(pipeline, output_model, compress=3)
    logger.info(f"\nModel saved to {output_model}")
    
    if nested_cv:
        return pipeline, feature_importances, {
            'cv_metrics': avg_metrics,
            'test_metrics': final_metrics
        }
    else:
        return pipeline, feature_importances, final_metrics

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
    geometry_pocket_df, geometry_residue_df = get_geometry_features(pdb_file)
    features_df = integrate_geometry_with_ml_features(features_df, geometry_residue_df)
    features_df = engineer_additional_features(features_df)
    
    # Load the model
    rf = joblib.load(model_file)

    # Drop non-feature columns
    feature_columns = features_df.columns.tolist()
    non_feature_cols = ['pdb_id', 'chain_id', 'residue_id', 'residue_name', 'ss_type', 'ss_ ', 'ss_ _asa', 'ss_ _hydro', 'ss_X_asa', 'ss_type_asa', 'ss_type_hydro']
    feature_columns = [col for col in feature_columns if col not in non_feature_cols]

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
    logger.info(f"Predictions saved to {output_file}")

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
    process_parser.add_argument("--log_file", help="Path to log file. If not provided, logs only to console.")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train an optimized model")
    train_parser.add_argument("--features_file", required=True, help="CSV file with labeled features")
    train_parser.add_argument("--output_model", default="binding_site_model.pkl", help="Output model file")
    train_parser.add_argument("--nested_cv", action="store_true", help="Use nested cross-validation")
    train_parser.add_argument("--log_file", help="Path to log file. If not provided, logs only to console.")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run predictions on a new PDB file")
    predict_parser.add_argument("--pdb_file", required=True, help="PDB file to predict on")
    predict_parser.add_argument("--model_file", default="binding_site_model.pkl", help="Trained model file")
    predict_parser.add_argument("--output_dir", default="predictions", help="Directory to save predictions")
    predict_parser.add_argument("--log_file", help="Path to log file. If not provided, logs only to console.")
    
    args = parser.parse_args()
    
    # Set up file logging if log_file is provided
    if hasattr(args, 'log_file') and args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {args.log_file}")
    
    # Run the appropriate command
    if args.command == "process":
        create_labeled_dataset(args.pdb_dir, args.prediction_dir, args.output_dir)
    elif args.command == "train":
        train_optimized_model(args.features_file, args.output_model, nested_cv=args.nested_cv)
    elif args.command == "predict":
        run_prediction_pipeline(args.pdb_file, args.model_file, args.output_dir)
    else:
        logger.error("No command specified. Use -h for help.")
        parser.print_help()
