# train_model.py
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(training_data_file, labels_file, model_file):
    training_data = np.load(training_data_file)
    labels = np.load(labels_file)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(training_data, labels)
    joblib.dump(model, model_file)
    print(f"Model trained and saved to {model_file}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python train_model.py <training_data> <labels> <model_file>")
        sys.exit(1)
    
    training_data_file, labels_file, model_file = sys.argv[1], sys.argv[2], sys.argv[3]
    train_model(training_data_file, labels_file, model_file)

if __name__ == "__main__":
    main()

# predict_binding.py
import sys
import numpy as np
import joblib
from Bio import PDB

def extract_features(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    feature_list = []
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):  # Ensure it's an amino acid
                    residues.append(residue)
                    ca = residue["CA"] if "CA" in residue else None
                    if ca:
                        feature_list.append([ca.coord[0], ca.coord[1], ca.coord[2]])
    
    return np.array(feature_list), residues

def predict_binding_sites(pdb_file, model_file):
    features, residues = extract_features(pdb_file)
    model = joblib.load(model_file)
    predictions = model.predict(features)
    
    binding_sites = [res for res, pred in zip(residues, predictions) if pred == 1]
    
    return binding_sites

def save_predictions(pdb_file, binding_sites, output_file):
    with open(pdb_file, 'r') as pdb, open(output_file, 'w') as out:
        for line in pdb:
            if any(str(res.id[1]) in line for res in binding_sites):
                out.write(f"{line[:60]} BIND\n")  # Mark binding sites
            else:
                out.write(line)

def main():
    if len(sys.argv) != 4:
        print("Usage: python predict_binding.py <pdb_file> <model_file> <output_file>")
        sys.exit(1)
    
    pdb_file, model_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    binding_sites = predict_binding_sites(pdb_file, model_file)
    save_predictions(pdb_file, binding_sites, output_file)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
