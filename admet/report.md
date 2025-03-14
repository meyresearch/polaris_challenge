# Polaris ASAP - ADMET competition

This is report to accompany our submission for the ADMET challenge. 

## 1. Adding in censored observations
Files: `missing_data_2.ipynb`

Values which were outside the observation range (represented as `NaN` in the training data) were included in the training data from inspection of the raw data package. Their values were recorded as the observation limit (e.g., for KSOL values recorded as `>400` in the raw data were recorded as `400`). In addition, the HLM and MLM targets were all censored to a lower limit of `10` following discussions with the competition team. 


## 2. Data splitting
Files: 
- `../utils/splitting_max_min.ipynb`
- `data/train_split2_idx.txt`
- `data/val_split2_idx.txt`


The training data were further split into a training and validation split.  This was done by finding the minimum tanimoto distance (`d`, based on  Morgan fingerprints) between each training instance and the test set.  Each training instance was placed in the validation set according to a Bernooulli process with probability `p = exp(-ld)` where `l` was tuned to give a 80/20 training/validation split. This ensured the validation set was similar to the test set, while a small number training instances were also similar to the test set. This function has been lost but the actual splits can be found in the files listed above. 


## 3. Target transformation
files `dm_features/features.ipynb`
The target values (except LogD) were transformed  according to `log(y + 1e-6)`

## 4. Imputing missing target values
Files: `dm_features/MICE.R`

I assumed that the missing target values were 'Missing At Random' and so suitable for imputation. Multiple Imputation by Chained Equations (MICE) using the `MICE` (https://cran.r-project.org/web/packages/mice/index.html) package in `R`. Only the target values were used as predictors, with 20 iterations, using predictive mean matching. 100 imputed datasets were created. 

## 5. Modelling
Files: 

- `deep_ord/fit_optimized_model.ipynb`


### Target transformation
Files: `deep_ord/utils:digitize_y`
The target values were transformed into an ordinal response using the unique training values (the 'train' split of the original training values) as the class boundaries. This allowed the inclusion of the censored target values (see for example Regression Modeling Strategies, FE Harrel Jr https://hbiostat.org/rmsc/, chapters 14 and 15). 

 A multitask deep learning ordinal response model was written in a fork of the SpaceCutter package (original [here](https://github.com/EthanRosenthal/spacecutter) and the multitask version [here](https://github.com/RobertArbon/spacecutter/tree/mtl/spacecutter)).  The univariate loss function for each target was the negative loglikelihood of logistic cumulative link loss (aka a Logistic ordinal response / proportional odds model). The multivariate loss function was the weighted mean of the univariate loss function. The weights were proportional to the inverse of the number of classes. This was an ad-hoc correction to ensure the that losses for each target were weighted approximately equally.  

### Features
Files: `deep_ord/features.ipynb`
Two sets of features were tried: 
1. Chemberta: https://molfeat.datamol.io/featurizers/ChemBERTa-77M-MTR
2. ChemProp: (via AdmetAI) https://github.com/swansonk14/admet_ai
the combination of  features were also tried. 
See the notebook for the exact features used. 

### Model architecture
Files: `deep_ord/optimize.ipynb`
the model consisted of a common 'backbone' and 5 'head' modules (all of the same size and type) which were used to calculate the loss.  Each module consisted of fully connected layers with ReLU activation.  The backbone module width was the number of features. The head width was also the number of features, except that the final output dimension was 1. 

### Optimization
Files: `deep_ord/optimize.ipynb`
The MTL Ordinal Model was optimized using the Adam Optimizer with a variable weight decay. Early stopping was used with a patience of either 10 or 100 epochs.  The batch size was the number of training instances. 

The model hyperparameters (weight decay, backbone and head depth, patience and features) were optimized using Optuna with a Tree Parzen Optimizer.  Multiobjective optimization was used with the validation mean absolute error (MAE(val)) and the absolute difference between MAE(val) and MAE(train) as objectives. 

The optimization was performed on the first imputed dataset.  The validation set was not imputed so the validation MAE was calculated for the non-missing target values. 

The final optimized hyperparameters were: 
- ChembertA + ChemProp features
- Head and backbone depth of 1
- Weight decay of 9.9e-5
- Patience of 100. 

The trials were recorded and will be made available. 

## 6. Predictions. 
Files: `deep_ord/predictions.ipynb`
A model with the optimized hyperparameters was fit for each of the imputed datasets and a set of predictions made. The final set of predictions were the average over the different model predictions. 
