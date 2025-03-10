library("readr")
library("rms")
library('mice')
library('mvord')


imp <- readRDS('admet/dm_features/ordinal_data_split_2/train_admet_split2_log_pmm_imputed.RDS')
# val <- read.csv("admet/dm_features/train_admet_split2_features.csv")
# val <- val[val$split == "val",]
# train <- val[val$split == "train",]
imp_list <- complete(imp, action='all', include=F)
predictors <- c('clogp', 'fsp3', 'mw', 'n_aliphatic_carbocycles', 'n_aliphatic_heterocyles', 'n_aliphatic_rings', 'n_aromatic_carbocycles', 'n_aromatic_heterocyles', 'n_aromatic_rings', 'n_heavy_atoms', 'n_hetero_atoms', 'n_lipinski_hba', 'n_lipinski_hbd', 'n_rings', 'n_rotatable_bonds', 'n_saturated_carbocycles', 'n_saturated_heterocyles', 'n_saturated_rings', 'sas', 'tpsa')

targets <- c("LogHLM", "LogMLM", "LogKSOL", "LogMDR1.MDCKII", "LogD")




df <- imp_list[[1]]
ord_targets <- sapply(targets, function(x) paste0("ord", x))

for (i in seq(1, length(targets))){
  x <- df[[targets[[i]]]]
  y <- ord_targets[[i]]
  df[[y]] <- factor(x, levels=sort(unique(x)), ordered=T)
}
formula <- paste0("MMO2(", paste(ord_targets, collapse=","), ")~", paste(predictors, collapse=" + "))
res <- mvord(formula=as.formula(formula), 
            data=df)