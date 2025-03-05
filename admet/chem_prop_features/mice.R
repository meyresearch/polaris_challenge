require(mice)
require(lattice)
require(ggmice)
library(ggplot2)
library(readr)

seed <- 1234


allVars <- c("split", "Molecule.Name", "CXSMILES", 
'LogD',
 'LogMLM',
 'LogHLM',
 'LogKSOL',
 'LogMDR1.MDCKII',
'BCUT2D_LOGPLOW',
 'BCUT2D_MRHI',
 'BCUT2D_MRLOW',
 'BCUT2D_MWHI',
 'HallKierAlpha',
 'Lipophilicity_AstraZeneca',
 'MaxAbsPartialCharge',
 'MaxPartialCharge',
 'MinAbsPartialCharge',
 'MinPartialCharge',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAmideBonds',
 'NumHDonors',
 'NumHeterocycles',
 'PAMPA_NCATS',
 'PEOE_VSA1',
 'PEOE_VSA10',
 'PEOE_VSA11',
 'SMR_VSA10',
 'SMR_VSA5',
 'SMR_VSA6',
 'SMR_VSA7',
 'SPS',
 'SlogP_VSA10',
 'SlogP_VSA3',
 'SlogP_VSA6',
 'VSA_EState1',
 'VSA_EState3',
 'VSA_EState9',
 'fr_C_O',
 'fr_C_O_noCOO',
 'fr_amide',
 'fr_aniline',
 'qed') 

df <- read.csv("admet/chem_prop_features/train_admet_features_1.csv")

df <- df[df$split == "train", allVars]



imputerVars <- c('LogMLM', 'LogHLM', 'LogKSOL', 'LogD', 'LogMDR1.MDCKII')
nonImputeVars <- c(
"split", "Molecule.Name", "CXSMILES",
'BCUT2D_LOGPLOW',
 'BCUT2D_MRHI',
 'BCUT2D_MRLOW',
 'BCUT2D_MWHI',
 'HallKierAlpha',
 'Lipophilicity_AstraZeneca',
 'MaxAbsPartialCharge',
 'MaxPartialCharge',
 'MinAbsPartialCharge',
 'MinPartialCharge',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAmideBonds',
 'NumHDonors',
 'NumHeterocycles',
 'PAMPA_NCATS',
 'PEOE_VSA1',
 'PEOE_VSA10',
 'PEOE_VSA11',
 'SMR_VSA10',
 'SMR_VSA5',
 'SMR_VSA6',
 'SMR_VSA7',
 'SPS',
 'SlogP_VSA10',
 'SlogP_VSA3',
 'SlogP_VSA6',
 'VSA_EState1',
 'VSA_EState3',
 'VSA_EState9',
 'fr_C_O',
 'fr_C_O_noCOO',
 'fr_amide',
 'fr_aniline',
 'qed')

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

write_csv(imp_df, "admet/chem_prop_features/train_admet_log_pmm_imputed.csv")
saveRDS(imp, "admet/chem_prop_features/train_admet_log_pmm_imputed.RDS")

