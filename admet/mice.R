require(mice)
require(lattice)
require(ggmice)
library(ggplot2)
library(readr)

seed <- 1234



df <- read.csv('admet/imputed/val_admet_all_transformed.csv')
print(str(df))
allVars <- c('CXSMILES', 'Molecule.Name',
            'LogMLM', 'LogHLM', 'LogKSOL', 'LogD', 'LogMDR1.MDCKII')

numDataSets <- 100
nIterations <- 20

# This specifies what columns should be used to predict what.  By default, the 'CXSMILES', and 'Molecule.Name' 
# columns will be used.  
predictorMatrix <- matrix(0, ncol = length(allVars), nrow = length(allVars))
rownames(predictorMatrix) <- allVars
colnames(predictorMatrix) <- allVars
imputerVars <- c('LogMLM', 'LogHLM', 'LogKSOL', 'LogD', 'LogMDR1.MDCKII')
predictorMatrix[, imputerVars] <- 1
diag(predictorMatrix) <- 0
predictorMatrix[c('CXSMILES', 'Molecule.Name'), ] <- 0


# Using PMM throghout
imp <- mice(data = df[allVars],
            m = numDataSets,
            method = c("", "", "pmm", "pmm", "pmm", "pmm", "pmm"),
            predictorMatrix = predictorMatrix,
            visitSequence = 'monotone', 
            maxit = nIterations, 
            seed = seed)

imp_df <- complete(imp, 'long', include=TRUE)

write_csv(imp_df, 'admet/imputed/val_admet_log_pmm.csv')
saveRDS(imp, 'admet/imputed/val_admet_log_pmm.RDS')

