library("readr")
library("rms")
library('mice')


imp <- readRDS('admet/chem_prop_features/train_admet_log_pmm_imputed.RDS')
val <- read.csv("admet/chem_prop_features/train_admet_features_1.csv")
val <- val[val$split == "val",]
imp_list <- complete(imp, action='all', include=F)
# imp_df <- complete(imp, "long", include=True)
# for (target in c("LogMLM", "LogHLM", "LogKSOL", "LogMDR1.MDCKII")){
#     x <- imp_df[, target]
#     bins <- c(0, unique(sort(x)))
#     imp_df[, paste0("ord", target)] <- cut(x, breaks=bins, label=F, include.lowest=F, ordered_result=T)
# }
# imp <- as.mids(imp_df)

var_by_target = list(LogD = c('fr_C_O',
  'fr_C_O_noCOO',
  'NumAliphaticHeterocycles',
  'NumAliphaticRings',
  'NumHDonors',
  'fr_aniline',
  'NumHeterocycles',
  'fr_amide',
  'Lipophilicity_AstraZeneca',
  'NumAmideBonds'),
 LogMLM = c('MinAbsPartialCharge',
  'MaxPartialCharge',
  'BCUT2D_MRLOW',
  'SMR_VSA5',
  'VSA_EState3',
  'SlogP_VSA6',
  'SlogP_VSA10',
  'PEOE_VSA10',
  'BCUT2D_MWHI',
  'SMR_VSA7'),
  LogHLM = c('SlogP_VSA10',
  'BCUT2D_MWHI',
  'MinAbsPartialCharge',
  'MaxPartialCharge',
  'BCUT2D_LOGPLOW',
  'BCUT2D_MRHI',
  'MaxAbsPartialCharge',
  'VSA_EState1',
  'MinPartialCharge',
  'PEOE_VSA11'),
 LogKSOL = c('BCUT2D_LOGPLOW',
  'SlogP_VSA3',
  'BCUT2D_MRHI',
  'PEOE_VSA10',
  'SPS',
  'PEOE_VSA1',
  'MinAbsPartialCharge',
  'MaxPartialCharge',
  'MaxAbsPartialCharge',
  'BCUT2D_MWHI'),
 LogMDR1.MDCKII = c('BCUT2D_MRHI',
  'PEOE_VSA1',
  'SMR_VSA10',
  'VSA_EState9',
  'qed',
  'MaxPartialCharge',
  'HallKierAlpha',
  'PAMPA_NCATS',
  'MinAbsPartialCharge',
  'SMR_VSA6'))



targets <- c("LogHLM", "LogMLM", "LogKSOL", "LogMDR1.MDCKII", "LogD")
alphas <- seq(0.1, 100, by=0.5)

results <- data.frame(target = character(), 
                      alpha = double(), 
                      mae = double())

avg.predict <- function(df_list, form, alpha, predictors, target, val){
  fit_predict <- function(df){
    mod <- orm(data=df, formula=as.formula(form), family='logistic', penalty=alpha)
    val_X <- val[, predictors]
    val_y <- val[, target]
    cc <- complete.cases(val_y)
    val_X <- val_X[cc, ]
    val_y <- val_y[cc]
    y_hat <- predict(mod, newdata = val_X)
    mean(abs(y_hat - val_y))
  }
  
  median(sapply(df_list, fit_predict))
}

for (i in seq(1, length(targets))){
  
    target <- targets[[i]]
    predictors <- var_by_target[[target]]
    form <- paste(target, "~", paste(predictors, collapse = "+"))
    for (alpha in alphas){
        mae <- avg.predict(imp_list, form, alpha, predictors, target, val) 
        results[nrow(results)+1, ] <- c(target, alpha, mae)
    }
}
# formula <-
# write.csv(results, 'admet/chem_prop_features/single_task_logistic_ordinal.csv', row.names=FALSE)
