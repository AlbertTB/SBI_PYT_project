# Binding Site Prediction Tutorial
This document explains how to run the prediction pipeline.

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

1) Make sure you have the following installed

```
pip install biopython scikit-learn pandas numpy tqdm matplotlib scikit-image imbalanced-learn joblib
```

2) Example of folder structure for ease of use:
```
your_project/
├── pdbs/                  # Raw PDB files
├── predictions/           # CSVs containing binding site labels
├── labeled_features/      # Auto-created: labeled features from processing
├── predictions_output/    # Auto-created: predictions for new inputs
├── models/                # Trained models
```
3) Predict binding sites
```
python main.py predict --pdb_file pdbs/new_protein.pdb --model_file models/name_of_model --output_dir predictions/ --log_file path/to/log_file.log
```
This will generate:

- A `.csv` file with prediction

- An annotated PDB file (optionally used for PyMOL or Chimera visualization) 

