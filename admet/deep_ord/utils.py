# from math import isnan
import pandas as pd
import numpy as np
import torch
# from torch import nn
import datamol as dm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler

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


def to_model_format(train, val):
    """
    Puts the training and validation data into a convenient form. 
    """
    config = {}
    train_X = train[0]
    targets = list(train[1].keys())
    targets.sort()
    train_y = np.concatenate([train[1][target]['values'].reshape(-1, 1) for target in targets], axis=1)
    n_tasks = train_y.shape[1]
    n_classes_per_task = [np.unique(train_y[:, i]).shape[0] for i in range(n_tasks)]
    n_obs, n_features = train_X.shape
    print(f" {n_tasks} tasks\n classes/task: {n_classes_per_task}\n features: {n_features}, obs: {n_obs}")

    val_y = np.concatenate([val[1][target]['values'].reshape(-1, 1) for target in targets], axis=1) 
    val_X = val[0]
    train_val_X = np.vstack([train_X, val_X]).astype(np.float32)
    train_val_y = np.vstack([train_y, val_y]).astype(np.float32)
    train_ix = np.arange(train_X.shape[0])
    val_ix = np.arange(train_X.shape[0], train_val_X.shape[0])

    config['n_features'] = n_features
    config['n_tasks'] = n_tasks
    config['n_classes_per_task'] = n_classes_per_task
    config['targets'] = targets
    
    return train_val_X, train_val_y, train_ix, val_ix, config

def ord_to_cont(train, y_ord):
    targets = list(train[1].keys())
    targets.sort() 
    y_cont = []
    for i, target in enumerate(targets):
        bins = targets[2][target]['bins']
        y_cont.apend(np.array([bins[x] if not np.isnan(x) else np.nan for x in y_ord[:, i]]).reshape(-1, 1))
    return np.concatenate(y_cont, axis=1)

def mtl_mae(train, y_pred, y_true_cont):
    y_pred_cont = ord_to_cont(train, y_pred)
    diff = np.abs(y_pred_cont - y_true_cont)
    return np.mean(diff, where=~np.isnan(diff))
    
def plot_results(train, val_y_pred, val_y_true_cont, train_y_pred, train_y_true_cont):
    val_y_pred_cont = ord_to_cont(train, val_y_pred)
    train_y_pred_cont = ord_to_cont(train, train_y_pred) 
    cols = sns.color_palette('colorblind')

    targets = list(train[1].keys())
    targets.sort()
    fig, axes = plt.subplots(len(targets), figsize=(6, 3*len(targets)))
    for i, ax in enumerate(axes):
        min_val = np.min((val_y_pred_cont.min(), val_y_true_cont.min(), train_y_pred_cont.min(), train_y_true_cont.min()))
        max_val = np.max((val_y_pred_cont.max(), val_y_true_cont.max(), train_y_pred_cont.max(), train_y_true_cont.max())) 
        ax.scatter(val_y_pred_cont, val_y_true_cont, label='validation', color=cols[0])
        ax.scatter(train_y_pred_cont, train_y_true_cont, label='train', color=cols[1])
        ax.plot([min_val, max_val], [min_val, max_val], label='y=x', color='black')

        ax.annotate(text=f"val MAE: {mtl_mae(train, val_y_pred, val_y_true_cont):4.2f}", xy=(0.1, 0.9))
        ax.set_title(targets[i])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    return axes


def predict(skorch_model, X):
    mod = skorch_model.module_
    mod.eval()
    y_pred_list = mod.forward(torch.as_tensor(X))

    y_pred_list = [x.cpu().detach().numpy() for x in y_pred_list]
    y_preds_ord = [np.argmax(x, axis=1) for x in y_pred_list]
    return y_preds_ord

