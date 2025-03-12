# Project Overview  

This repository is a work in progress and will be updated as I progress through the challenge.  

## Files and Directories  

- **`pred_test.ipynb`** – The main file where I experiment with the data and model.  
- **`reference_structures/`** – Contains reference complex PDB files for aligning test data.  
All training data is limited to the first 100 data points.
- **`train_complex_truth/`** – Contains complexes PDB of the training data provided by the challenge.
- **`train_complex_predicted/`** – Contains complexes PDB of the training data predicted by Boltz-1
- **`train_ligand_output/`** – Extracted Predicted ligand in SDFs from *train_complex_predicted*. 
- **`train_ligand_bond_fixed/`** – Contains Boltz-1 predicted ligands with bond order fixed using MolReBond.
  
- Its missing the 0th molecule, so 99 molecules effectively. The first molecule has some issues.

We will release our test ligand prediction after the challenge ends in the repo. 