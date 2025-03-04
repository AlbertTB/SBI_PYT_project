import os
import random
import shutil

# Configuration - change these paths
dir1 = 'holo4k/receptor'          # Source folder (main files)
dir2 = 'holo4k/ligand'         # Folder with corresponding _lig.pdb files
dir3 = 'holo4k/predictions'             # Folder with corresponding .csv files

# Output folders
output_dir1 = 'train_data/holo4k/receptor'     # Output for selected files from dir1
output_dir2 = 'train_data/holo4k/ligand'     # Output for corresponding _lig.pdb files
output_dir3 = 'train_data/holo4k/predictions'     # Output for corresponding .csv files

# Target number of files to select
num_files_to_select = 1000

# Create output directories if they don't exist
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)
os.makedirs(output_dir3, exist_ok=True)

# Get all candidate files in dir1
all_files = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
random.shuffle(all_files)  # Shuffle to make random selection easier

# Track selected files
selected_files = []

# Loop until we have enough valid pairs or run out of files
for file_name in all_files:
    if len(selected_files) >= num_files_to_select:
        break

    base_name, ext = os.path.splitext(file_name)

    # Check for corresponding files
    lig_file = f"{base_name}_lig.pdb"
    csv_file = f"{base_name}.csv"

    lig_path = os.path.join(dir2, lig_file)
    csv_path = os.path.join(dir3, csv_file)

    # Only accept this file if both corresponding files exist
    if os.path.exists(lig_path) and os.path.exists(csv_path):
        selected_files.append(file_name)

# Copy files once we have the valid list
for file_name in selected_files:
    base_name, ext = os.path.splitext(file_name)

    # --- Copy file from dir1 ---
    shutil.copy2(os.path.join(dir1, file_name), os.path.join(output_dir1, file_name))

    # --- Copy corresponding _lig.pdb file ---
    lig_file = f"{base_name}_lig.pdb"
    shutil.copy2(os.path.join(dir2, lig_file), os.path.join(output_dir2, lig_file))

    # --- Copy corresponding .csv file ---
    csv_file = f"{base_name}.csv"
    shutil.copy2(os.path.join(dir3, csv_file), os.path.join(output_dir3, csv_file))

print(f"Selected and copied {len(selected_files)} complete triplets (file, _lig.pdb, .csv).")

