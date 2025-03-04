#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import glob
from collections import defaultdict

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
            res_name = res_id_parts[2]
            res_num = int(res_id_parts[3])
            
            # Store the binding class (ignoring the prediction column as instructed)
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
    Create a labeled dataset by combining PDB features with binding site information.
    
    Parameters:
    -----------
    pdb_dir : str
        Directory containing PDB files
    prediction_dir : str
        Directory containing prediction files
    output_dir : str
        Directory to save the labeled dataset
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing combined features and labels
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Import the feature extraction module (assuming it's in the same directory)
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from pdb_feature_extraction import extract_features, process_directory
    
    # Process all prediction files
    binding_sites = process_all_prediction_files(prediction_dir)
    
    # Get PDB files that have prediction data
    pdb_files = []
    for pdb_id in binding_sites.keys():
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if os.path.exists(pdb_file):
            pdb_files.append(pdb_file)
        else:
            print(f"Warning: PDB file not found for {pdb_id}")
    
    # Extract features for each PDB file
    all_features = []
    for pdb_file in pdb_files:
        pdb_id = os.path.basename(pdb_file).split('.')[0]
        print(f"Processing {pdb_id}...")
        
        # Extract features
        features_df = extract_features(pdb_file, output_dir)
        
        # Add binding site labels
        features_df['is_binding_site'] = 0
        
        if pdb_id in binding_sites:
            for i, row in features_df.iterrows():
                chain_id = row['chain_id']
                res_id = int(row['residue_id'])
                
                if (chain_id, res_id) in binding_sites[pdb_id]:
                    features_df.at[i, 'is_binding_site'] = binding_sites[pdb_id][(chain_id, res_id)]
        
        # Save labeled features for this PDB
        output_file = os.path.join(output_dir, f"{pdb_id}_labeled_features.csv")
        features_df.to_csv(output_file, index=False)
        
        all_features.append(features_df)
    
    # Combine all features
    if all_features:
        combined_df = pd.concat(all_features, ignore_index=True)
        combined_output = os.path.join(output_dir, "combined_labeled_features.csv")
        combined_df.to_csv(combined_output, index=False)
        
        # Print class distribution
        binding = combined_df['is_binding_site'].sum()
        total = len(combined_df)
        print(f"\nBinding sites: {binding} ({binding/total*100:.2f}%)")
        print(f"Non-binding sites: {total-binding} ({(total-binding)/total*100:.2f}%)")
        
        return combined_df
    else:
        print("No features extracted.")
        return None

def convert_prediction_to_binding_sites_json(prediction_dir, output_file="binding_sites.json"):
    """
    Convert prediction files to a binding sites JSON file for the feature extraction script.
    
    Parameters:
    -----------
    prediction_dir : str
        Directory containing prediction files
    output_file : str
        Path to save the binding sites JSON file
        
    Returns:
    --------
    dict
        Dictionary mapping PDB IDs to binding site information
    """
    import json
    
    # Process all prediction files
    all_binding_sites = process_all_prediction_files(prediction_dir)
    
    # Convert to the format expected by the feature extraction script
    binding_sites_json = {}
    
    for pdb_id, sites in all_binding_sites.items():
        binding_sites_json[pdb_id] = []
        
        for (chain_id, res_id), binding_class in sites.items():
            if binding_class == 1:  # Only include positive binding sites
                binding_sites_json[pdb_id].append([chain_id, res_id])
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(binding_sites_json, f, indent=2)
    
    print(f"Binding sites JSON saved to {output_file}")
    return binding_sites_json

def train_random_forest_model(features_file, output_model="binding_site_model.pkl"):
    """
    Train a Random Forest model on the labeled dataset.
    
    Parameters:
    -----------
    features_file : str
        Path to the CSV file containing labeled features
    output_model : str
        Path to save the trained model
        
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    import joblib
    
    print(f"Training Random Forest model on {features_file}...")
    
    # Load the features
    df = pd.read_csv(features_file)
    
    # Drop non-feature columns
    feature_columns = df.columns.tolist()
    non_feature_cols = ['pdb_id', 'chain_id', 'residue_id', 'residue_name', 'ss_type', 'is_binding_site']
    
    for col in non_feature_cols:
        if col in feature_columns:
            feature_columns.remove(col)
    
    # Prepare features and target
    X = df[feature_columns].fillna(0)  # Replace NaN with 0
    y = df['is_binding_site']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train the model with class balancing
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train on the full training set
    rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Feature importance
    feature_importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importances.head(10))
    
    # Save the model
    joblib.dump(rf, output_model)
    print(f"\nModel saved to {output_model}")
    
    return rf

def run_prediction_pipeline(pdb_file, model_file="binding_site_model.pkl", output_dir="predictions"):
    """
    Run the prediction pipeline on a new PDB file.
    
    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file
    model_file : str
        Path to the trained model
    output_dir : str
        Directory to save the predictions
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing predicted binding sites
    """
    import joblib
    from pdb_feature_extraction import extract_features
    
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
        'class': 0,  # Placeholder as per the format you provided
        'prediction': features_df['prediction'],
        'probability': features_df['probability']
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prediction files and generate binding site features")
    
    # Command group
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process predictions command
    process_parser = subparsers.add_parser("process", help="Process prediction files and extract features")
    process_parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB files")
    process_parser.add_argument("--prediction_dir", required=True, help="Directory containing prediction files")
    process_parser.add_argument("--output_dir", default="labeled_features", help="Directory to save labeled features")
    
    # Convert predictions to JSON command
    convert_parser = subparsers.add_parser("convert", help="Convert prediction files to binding sites JSON")
    convert_parser.add_argument("--prediction_dir", required=True, help="Directory containing prediction files")
    convert_parser.add_argument("--output_file", default="binding_sites.json", help="Output JSON file")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train a Random Forest model")
    train_parser.add_argument("--features_file", required=True, help="CSV file with labeled features")
    train_parser.add_argument("--output_model", default="binding_site_model.pkl", help="Output model file")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run predictions on a new PDB file")
    predict_parser.add_argument("--pdb_file", required=True, help="PDB file to predict on")
    predict_parser.add_argument("--model_file", default="binding_site_model.pkl", help="Trained model file")
    predict_parser.add_argument("--output_dir", default="predictions", help="Directory to save predictions")
    
    args = parser.parse_args()
    
    if args.command == "process":
        create_labeled_dataset(args.pdb_dir, args.prediction_dir, args.output_dir)
    elif args.command == "convert":
        convert_prediction_to_binding_sites_json(args.prediction_dir, args.output_file)
    elif args.command == "train":
        train_random_forest_model(args.features_file, args.output_model)
    elif args.command == "predict":
        run_prediction_pipeline(args.pdb_file, args.model_file, args.output_dir)
    else:
        parser.print_help()