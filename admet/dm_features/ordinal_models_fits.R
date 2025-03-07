library("readr")
library("rms")
library('mice')


imp <- readRDS('admet/dm_features/train_admet_split2_log_pmm_imputed.RDS')
val <- read.csv("admet/dm_features/train_admet_split2_features.csv")
val <- val[val$split == "val",]
imp_list <- complete(imp, action='all', include=F)
predictors <- c('clogp', 'fsp3', 'mw', 'n_aliphatic_carbocycles', 'n_aliphatic_heterocyles', 'n_aliphatic_rings', 'n_aromatic_carbocycles', 'n_aromatic_heterocyles', 'n_aromatic_rings', 'n_heavy_atoms', 'n_hetero_atoms', 'n_lipinski_hba', 'n_lipinski_hbd', 'n_rings', 'n_rotatable_bonds', 'n_saturated_carbocycles', 'n_saturated_heterocyles', 'n_saturated_rings', 'sas', 'tpsa')

targets <- c("LogHLM", "LogMLM", "LogKSOL", "LogMDR1.MDCKII", "LogD")


avg.predict <- function(df_list, form, alpha, predictors, target, val){

  fit_predict <- function(df, val_X){
    mod <- orm(data=df, formula=as.formula(form), family='logistic', penalty=alpha)
    y_hat <- predict(mod, newdata = val_X)
    results <- list(predictions = y_hat, coefficients = mod$coefficients[predictors])
    results
  }

  # get raw val data
  val_X <- val[, predictors]
  val_y <- val[, target]
  # subset non-missing
  cc <- complete.cases(val_y)
  val_X <- val_X[cc, ]
  val_y <- val_y[cc]
  # make predictions over imputed datasets
  results <- lapply(df_list, fit_predict, val_X)
  # preds <- sapply(results$)
  preds <- lapply(results, function(x) x['predictions'][[1]])
  betas <- lapply(results, function(x) x['coefficients'][[1]])

  # # take average of all predictions
  avg_yhat <- colMeans(do.call(rbind, preds))
  avg_beta <- colMeans(do.call(rbind, betas))
  list(val_y_hat = avg_yhat, beta = avg_beta, val_y=val_y)
}


predict_mae <- function(df_list, form, alpha, predictors, target, val){
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





alphas <- list(
LogD	 = 100, 
	LogHLM	= 4.444444, 
LogKSOL =	6.666667, 
LogMDR1.MDCKII =	100, 
LogMLM =	0.489796
         )
for (target in targets){
  form <- paste(target, "~", paste(predictors, collapse = "+"))
  alpha <- alphas[[target]]

  results <- avg.predict(imp_list, form, alpha, predictors, target, val)
  print(mean(abs(results$val_y_hat - results$val_y)))
  preds <- data.frame(yhat = results$val_y_hat, y=results$val_y)
  coefs <- data.frame(beta = results$beta)
  write.csv(preds, paste0("admet/dm_features/", target, "_split2_predictions.csv", collapse=""))
  write.csv(coefs, paste0("admet/dm_features/", target, "_split2_betas.csv", collapse=""))
}



