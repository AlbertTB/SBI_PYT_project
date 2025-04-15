import glob

pdb_files = glob.glob("test/astex/receptor/*.pdb")

for pdb_file in pdb_files:
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    
    # Check if HEADER exists
    if not lines[0].startswith("HEADER"):
        lines.insert(0, "HEADER    DUMMY PDB FILE\n")
    
    # Check if END exists
    if not any(line.startswith("END") for line in lines):
        lines.append("END\n")
    
    with open(pdb_file, "w") as f:
        f.writelines(lines)

    print(f"Fixed {pdb_file}")