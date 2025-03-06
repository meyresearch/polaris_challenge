library("readr")
library("rms")
library('mice')


imp <- readRDS('polaris_challenge/admet/dm_features/train_admet_split2_log_pmm_imputed.RDS')
df <- read.csv("polaris_challenge/admet/dm_features/train_admet_split2_features.csv")
val <- df[df$split == "val",]


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


imp_list <- complete(imp, action='all', include=F)
predictors <- c('clogp', 'fsp3', 'mw', 'n_aliphatic_carbocycles', 'n_aliphatic_heterocyles', 'n_aliphatic_rings', 'n_aromatic_carbocycles', 'n_aromatic_heterocyles', 'n_aromatic_rings', 'n_heavy_atoms', 'n_hetero_atoms', 'n_lipinski_hba', 'n_lipinski_hbd', 'n_rings', 'n_rotatable_bonds', 'n_saturated_carbocycles', 'n_saturated_heterocyles', 'n_saturated_rings', 'sas', 'tpsa')

# cont_preds <- c('clogp', 'fsp3', 'mw', 'sas', 'tpsa', 'n_rotatable_bonds') 
# ord_preds <- c(
#  ) 
# int_preds <- c('n_heavy_atoms', 'n_hetero_atoms',  'n_lipinski_hba', 
#  'n_lipinski_hbd',   'n_aliphatic_carbocycles', 
# 'n_aliphatic_heterocyles', 
# 'n_aliphatic_rings', 
# 'n_aromatic_carbocycles', 
# 'n_aromatic_heterocyles', 
# 'n_aromatic_rings', 
#  'n_rings', 
#  'n_saturated_carbocycles', 
#  'n_saturated_heterocyles', 
#  'n_saturated_rings')



alphas_by_target <- list(
  'LogHLM' = seq(0, 20, length.out=10), 
  'LogMLM' =seq(0, 1, length.out=50),  
  'LogKSOL' = seq(0, 20, length.out=10), 
  'LogD' = seq(0, 100, length.out=10), 
  'LogMDR1.MDCKII' = seq(0, 100, length.out=10)
)

targets <- c(
  # "LogHLM", 
  "LogMLM"
  # "LogKSOL"
  # "LogMDR1.MDCKII", 
  # "LogD"
  )

colMax <- function(data) sapply(data, max, na.rm = TRUE)
colMin <- function(data) sapply(data, min, na.rm = TRUE)

# ord_pred_max <- colMax(df[, ord_preds])

# ord_pred_levels <- lapply(ord_pred_max, function(x) paste0("c(", paste0(seq(0, x),collapse=","), ")"))

# cont_predictor_form <- paste(sapply(cont_preds, function(x) paste0("rcs(", x, ", 3)")), collapse="+")

# ord_pred_form <- paste(sapply(seq(1, length(ord_preds)), function(x) paste0("scored(", ord_preds[[x]], ",", ord_pred_levels[[x]], ")")), collapse="+")

int_preds_form <- paste(int_preds, collapse="+")

# predictor_form <- paste(c(cont_predictor_form, int_preds_form), collapse="+")
predictor_form <- paste(predictors, collapse="+")

for (i in seq(1, length(targets))){
  
    target <- targets[[i]]
    print(target)
    form <- paste(target, "~", predictor_form)
    alphas <- alphas_by_target[[target]]

    for (alpha in alphas){
        tryCatch(
          {
            mae <- avg.predict(imp_list, form, alpha, predictors, target, val) 
            print(c(alpha, mae))
            results[nrow(results)+1, ] <- c(target, alpha, mae)
          }, 
          error = function(cond){
            message(paste("model not fit with:", alpha))

          }
        )
    }
}
write.csv(results, 'polaris_challenge/admet/dm_features/single_task_logistic_ordinal_scan_5.csv', row.names=FALSE)

