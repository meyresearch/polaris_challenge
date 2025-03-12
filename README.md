# Offical Repository of Meyresearch for the ASAP-Polaris Challenge 
This repository contains the code and data used for the ASAP-Polaris Challenge by Meyresearch.


   **env.yml** - minimal environment for downloading dependencies for the Challenge.

## Pose
by @AuroVarat

The Pose Prediction is using Boltz-1. Please refer to the [offical Boltz-1 repository](https://github.com/jwohlwend/boltz) for more information on how to install. 
  
The next important aspect of the project is replacing or adding bond order information from the reference SMILES to the predicted structure by Boltz-1. Boltz-1 does not take into accont bond order information and does not output it in the predicted structure. This is solved by using [MolReBond](https://github.com/AuroVarat/MolReBond) that I have made.
There is likely better ways to do this.
  
Please find the structure in the README inside the submodule.


## ADMET
`eda.ipynb` some basic EDA on the admet data.