library("readr")
library("rms")
library('mice')


imp <- readRDS('polaris_challenge/admet/dm_features/train_admet_log_pmm_imputed.RDS')
val <- read.csv("polaris_challenge/admet/dm_features/train_admet_features.csv")
val <- val[val$split == "val",]
imp_list <- complete(imp, action='all', include=F)
predictors <- c('clogp', 'fsp3', 'mw', 'n_aliphatic_carbocycles', 'n_aliphatic_heterocyles', 'n_aliphatic_rings', 'n_aromatic_carbocycles', 'n_aromatic_heterocyles', 'n_aromatic_rings', 'n_heavy_atoms', 'n_hetero_atoms', 'n_lipinski_hba', 'n_lipinski_hbd', 'n_rings', 'n_rotatable_bonds', 'n_saturated_carbocycles', 'n_saturated_heterocyles', 'n_saturated_rings', 'qed', 'sas', 'tpsa')



targets <- c("LogHLM", "LogMLM", "LogKSOL", "LogMDR1.MDCKII", "LogD")


results <- data.frame(target = character(), 
                      alpha = double(), 
                      mae = double())

avg.predict <- function(df_list, form, alpha, predictors, target, val){
  fit_predict <- function(df, val_X){
    mod <- orm(data=df, formula=as.formula(form), family='logistic', penalty=alpha)
    y_hat <- predict(mod, newdata = val_X)
    y_hat
  }
  # get raw val data
  val_X <- val[, predictors]
  val_y <- val[, target]
  # subset non-missing
  cc <- complete.cases(val_y)
  val_X <- val_X[cc, ]
  val_y <- val_y[cc]
  # make predictions over imputed datasets
  preds <- lapply(df_list, fit_predict, val_X)
  # take average of all predictions
  avg_yhat <- colMeans(do.call(rbind, preds))
  mae <- mean(abs(avg_yhat - val_y))
  mae
}

alphas_by_target <- list(
  'LogHLM' = seq(1, 5, length.out=10), 
  'LogMLM' =seq(1, 5, length.out=10),  
  'LogKSOL' = seq(0, 1, length.out=10), 
  'LogD' = seq(40, 80, length.out=10), 
  'LogMDR1.MDCKII' = seq(125, 150, length.out=10)
)
targets <- c('LogHLM')

for (i in seq(1, length(targets))){
  
    target <- targets[[i]]
    print(target)
    form <- paste(target, "~", paste(predictors, collapse = "+"))
    alphas <- alphas_by_target[[target]]

    for (alpha in alphas){
        tryCatch(
          {
            mae <- avg.predict(imp_list, form, alpha, predictors, target, val) 
            print(c(mae, alpha))
            results[nrow(results)+1, ] <- c(target, alpha, mae)
          }, 
          error = function(cond){
            message(paste("model not fit with:", alpha))

          }
        )
    }
}
# write.csv(results, 'polaris_challenge/admet/dm_features/single_task_logistic_ordinal_scan_6 .csv', row.names=FALSE)

