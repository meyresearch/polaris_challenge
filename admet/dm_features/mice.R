require(mice)
require(lattice)
require(ggmice)
library(ggplot2)
library(readr)

seed <- 1234

allVars <- c('split', 'clogp', 'fsp3', 'mw', 'n_aliphatic_carbocycles', 'n_aliphatic_heterocyles', 'n_aliphatic_rings', 'n_aromatic_carbocycles', 'n_aromatic_heterocyles', 'n_aromatic_rings', 'n_heavy_atoms', 'n_hetero_atoms', 'n_lipinski_hba', 'n_lipinski_hbd', 'n_rings', 'n_rotatable_bonds', 'n_saturated_carbocycles', 'n_saturated_heterocyles', 'n_saturated_rings', 'qed', 'sas', 'tpsa', 'LogD', 'LogMLM', 'LogHLM', 'LogKSOL', 'LogMDR1.MDCKII', 'Molecule.Name')
imputeVars <- c('LogD', 'LogMLM', 'LogHLM', 'LogKSOL', 'LogMDR1.MDCKII')
nonImputeVars <- c('split', 'Molecule.Name', 'clogp', 'fsp3', 'mw', 'n_aliphatic_carbocycles', 'n_aliphatic_heterocyles', 'n_aliphatic_rings', 'n_aromatic_carbocycles', 'n_aromatic_heterocyles', 'n_aromatic_rings', 'n_heavy_atoms', 'n_hetero_atoms', 'n_lipinski_hba', 'n_lipinski_hbd', 'n_rings', 'n_rotatable_bonds', 'n_saturated_carbocycles', 'n_saturated_heterocyles', 'n_saturated_rings', 'qed', 'sas', 'tpsa')



df <- read.csv("polaris_challenge/admet/dm_features/train_admet_split2_features.csv")
df <- df[df$split == "train", allVars]

numDataSets <- 100
nIterations <- 20

# This specifies what columns should be used to predict what.  By default, the 'CXSMILES', and 'Molecule.Name' 
# columns will be used.  
predictorMatrix <- matrix(0, ncol = length(allVars), nrow = length(allVars))
rownames(predictorMatrix) <- allVars
colnames(predictorMatrix) <- allVars


predictorMatrix[, imputerVars] <- 1
diag(predictorMatrix) <- 0

predictorMatrix[nonImputeVars, ] <- 0


# Using PMM throghout
imp <- mice(data = df[,allVars],
            m = numDataSets,
            predictorMatrix = predictorMatrix,
            visitSequence = "monotone",
            maxit = nIterations,
            seed = seed)

imp_df <- complete(imp, "long", include=TRUE)

write_csv(imp_df, "polaris_challenge/admet/dm_features/train_admet_split2_log_pmm_imputed.csv")
saveRDS(imp, "polaris_challenge/admet/dm_features/train_admet_split2_log_pmm_imputed.RDS")

