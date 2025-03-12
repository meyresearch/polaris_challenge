# from math import isnan
import pandas as pd
import numpy as np
# from spacecutter.models import OrdinalLogisticModel
# import torch
# from torch import nn
import datamol as dm
# import matplotlib.pyplot as plt

# from skorch import NeuralNet
# from skorch.dataset import Dataset
# from skorch.helper import SkorchDoctor
# from skorch.callbacks import EarlyStopping

# from spacecutter.callbacks import AscensionCallback
# from spacecutter.losses import CumulativeLinkLoss
# from sklearn.metrics import mean_absolute_error
# from scipy.stats import kendalltau

from sklearn.preprocessing import RobustScaler

# from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer


def digitize_y(y_s, n_cuts=None, bins_by_target=None, remove_nans=True):
    results = {}
    targets = list(y_s.columns)
    for target in targets:
        y = y_s.loc[:, target].values
        not_missing_ix = ~np.isnan(y)

        y_no_miss = y[not_missing_ix]
        max_y = np.max(y_no_miss)
   
        if bins_by_target is None:
          if n_cuts is None:
              bins =list(np.unique(np.sort(y_no_miss)))
          else:
              step = int(np.unique(y_no_miss).shape[0]//n_cuts)
              bins = list(np.unique(np.sort(y_no_miss))[::step])
        else:
          bins = list(bins_by_target[target]['bins'])

        if bins[-1] != max_y:
          bins.append(max_y)
          bins.sort()

        if remove_nans:
           y = y[not_missing_ix]

        y_ord, bins = pd.cut(y, bins=bins, include_lowest=True, labels=False, retbins=True, ordered=True)
        assert y.shape[0] == y_ord.shape[0], f"{target} {y.shape[0]} {y_ord.shape[0]}" 

        results[target] = {'values': y_ord, 'bins': bins, 'not_missing_ix': not_missing_ix, 'original': y }

    return results

def get_Xy(df, ix, n_cuts,  bins_by_target=None, features=None, scalers_by_feature=None, proj_dir=None, remove_nans=True):
    if isinstance(features, str):
      features = [features]
    X_all = []

    if scalers_by_feature is None:
       scalers_by_feature = {}

    for feature in features:
      if feature == 'fp':
        print('using morgan fingerprints')
        smiles = df.loc[ix, 'CXSMILES'].values
        X = np.vstack([dm.to_fp(dm.to_mol(smi)) for smi in smiles])
      elif feature == 'rdkit_simple':
        print('using rdkit simple')
        Xraw = df.loc[ix, ['clogp', 'fsp3', 'mw', 'n_aliphatic_carbocycles',
          'n_aliphatic_heterocyles', 'n_aliphatic_rings',
          'n_aromatic_carbocycles', 'n_aromatic_heterocyles', 'n_aromatic_rings',
          'n_heavy_atoms', 'n_hetero_atoms', 'n_lipinski_hba', 'n_lipinski_hbd',
          'n_rings', 'n_rotatable_bonds', 'n_saturated_carbocycles',
          'n_saturated_heterocyles', 'n_saturated_rings', 'qed', 'sas', 'tpsa']].values

        if feature in scalers_by_feature:
          print('\tusing existing scaler')
          scaler = scalers_by_feature[feature]
          X  = scaler.transform(Xraw)
        else:
          print('\tcreating new scaler')
          scaler = RobustScaler()
          X = scaler.fit_transform(Xraw)
          scalers_by_feature[feature] = scaler

      elif feature == 'chem_prop':
        print('using chemprop features')
        chem_prop_features = pd.read_csv(f'{proj_dir}/chem_prop_features/train_admet_chemprop.csv')
        Xdf = df.loc[ix, :].merge(chem_prop_features, on='Molecule.Name', how='left')
        Xraw = Xdf.loc[:, chem_prop_features.columns.difference(['Molecule.Name', 'split'])]
        if feature in scalers_by_feature:
          print('\tusing existing scaler')
          scaler = scalers_by_feature[feature]
          X  = scaler.transform(Xraw)
        else:
          print('\tcreating new scaler')
          scaler = RobustScaler()
          X = scaler.fit_transform(Xraw)
          scalers_by_feature[feature] = scaler
      elif feature == 'chemberta':
        print('using chemberta')
        chemberta = pd.read_csv(f'{proj_dir}/deep_ord/chemberta_77M_mtr.csv')
        Xdf = df.loc[ix, :].merge(chemberta, on='Molecule.Name', how='left')
        Xraw= Xdf.loc[:, chemberta.columns.difference(['Molecule.Name'])]
        if feature in scalers_by_feature:
          print('\tusing existing scaler')
          scaler = scalers_by_feature[feature]
          X  = scaler.transform(Xraw)
        else:
          print('\tcreating new scaler')
          scaler = RobustScaler()
          X = scaler.fit_transform(Xraw)
          scalers_by_feature[feature] = scaler
      else:
        raise ValueError(f"Unknown features: {feature}")
      X_all.append(X)
    X = np.hstack(X_all)

    ys = df.filter(regex='^Log').loc[ix, :]
    assert ys.shape[0] == X.shape[0]
    y_dig = digitize_y(ys, n_cuts=n_cuts, bins_by_target=bins_by_target, remove_nans=remove_nans)
    return X, y_dig, scalers_by_feature

def train_data(df_train, imp_ix, df_val, n_cuts=None, features='fp', proj_dir=None, remove_nans=True):
    ix_by_imp = None
    if imp_ix is None:
      train_ix = (df_train['split'] == 'train' ) & (df_train['.imp'] != 0)
      max_imp = df_train['.imp'].max()
      ix_by_imp = {i: df_train.loc[train_ix, '.imp'] == i for i in range(1, max_imp+1)}
    else:
      train_ix = (df_train['.imp'] == imp_ix) & (df_train['split'] == 'train')

    val_ix = df_val['split'] == 'val'

    results = []
    print('training data')
    X_train, y_train_by_targ, scalers_by_feature = get_Xy(df_train, train_ix, n_cuts, features=features, proj_dir=proj_dir, remove_nans=remove_nans)
    print('validation data')
    X_val, y_val_by_targ, _ = get_Xy(df_val, val_ix, n_cuts, y_train_by_targ, features=features, scalers_by_feature=scalers_by_feature, proj_dir=proj_dir, remove_nans=remove_nans)

    results.append((X_train, y_train_by_targ, ix_by_imp))
    results.append((X_val, y_val_by_targ))

    return results