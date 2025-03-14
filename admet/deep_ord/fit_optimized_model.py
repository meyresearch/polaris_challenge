#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import OrderedDict

import pandas as pd
import sklearn as sk
import numpy as np
import numpy as np

from torch import nn
import torch

from skorch import NeuralNet
from skorch.callbacks import EarlyStopping

from spacecutter.models import OrdinalLogisticMultiTaskModel
from spacecutter.losses import MultiTaskCumulativeLinkLoss
from spacecutter.callbacks import AscensionCallback

from utils import train_data, to_model_format, digitize_y


# In[2]:


proj_dir = '/Users/robertarbon/Library/CloudStorage/GoogleDrive-robert.arbon@gmail.com/My Drive/Polaris_ASAP_competition/polaris_challenge/admet'


# In[12]:


# Only contains the 'train' split (train_split2_idx.npy) of the polaris training data 
imputed_split2_data = pd.read_csv(f'{proj_dir}/dm_features/ordinal_data_split_2/train_admet_split2_log_pmm_imputed.csv')
# All the training data from polaris
all_training_data = pd.read_csv(f'{proj_dir}/dm_features/ordinal_data_split_2/train_admet_split2_features.csv')
# Test data from polaris. 
test_data = pd.read_csv(f"{proj_dir}/data/test_admet_all.csv")

# change names
imputed_split2_data.rename(columns={'Molecule Name': 'Molecule.Name', 'LogMDR1-MDCKII':'LogMDR1.MDCKII'}, inplace=True)
all_training_data.rename(columns={'Molecule Name': 'Molecule.Name', 'LogMDR1-MDCKII':'LogMDR1.MDCKII'}, inplace=True)
test_data.rename(columns={'Molecule Name': 'Molecule.Name', 'LogMDR1-MDCKII':'LogMDR1.MDCKII'}, inplace=True)

# Smiles columns because they were removed (for some unknown reason)
df_smiles = pd.read_csv(f'{proj_dir}/data/train_admet_all.csv')
df_smiles.rename(columns={'Molecule Name': 'Molecule.Name', 'LogMDR1-MDCKII':'LogMDR1.MDCKII'}, inplace=True)

imputed_split2_data = imputed_split2_data.merge(df_smiles.loc[:, ['Molecule.Name', 'CXSMILES']], on='Molecule.Name', how='left')
all_training_data = all_training_data.merge(df_smiles.loc[:, ['Molecule.Name', 'CXSMILES']], on='Molecule.Name', how='left')


train_features = pd.read_csv('train_features_by_molecule_name.csv')
test_features = pd.read_csv('test_features_by_molecule_name.csv')


# In[14]:


imputed_split2_data.shape[0], all_training_data.shape[0], test_data.shape[0]


# In[17]:


n_imputed_ds = imputed_split2_data['.imp'].unique().shape[0] # imp==0 is the original data. 
n_imputed_ds, imputed_split2_data.shape[0]/n_imputed_ds, (all_training_data['split']=='train').sum()


# In[28]:


target_cols = list(all_training_data.filter(regex='^Log'))
target_cols.sort()
target_cols


# In[20]:


cp_cols = [
    'BBB_Martins', 
    'Bioavailability_Ma',
    'CYP1A2_Veith',
    'CYP2C19_Veith',
    'CYP2C9_Substrate_CarbonMangels',
    'CYP2C9_Veith',
    'CYP2D6_Substrate_CarbonMangels',
    'CYP2D6_Veith',
    'CYP3A4_Substrate_CarbonMangels',
    'CYP3A4_Veith',
    'PAMPA_NCATS',
    'Pgp_Broccatelli',
    'Caco2_Wang',
    'Clearance_Hepatocyte_AZ',
    'Clearance_Microsome_AZ',
    'Half_Life_Obach',
    'HydrationFreeEnergy_FreeSolv',
    'Lipophilicity_AstraZeneca',
    'PPBR_AZ',
    'Solubility_AqSolDB',
    'VDss_Lombardo'
]
chemberta_cols = [str(x) for x in range(384)]


# In[19]:





# In[37]:


# get_ipython().system('export MKL_ENABLE_INSTRUCTIONS=SSE4_2')


# In[ ]:


# - features = 'chem_prop' + 'chemberta'
# - weight decay = 9.9e-5
# - backbone depth, head depth = 1, 1
weight_decay = 9.9e-5
backbone_depth, head_depth = 1, 1
patience = 100

predictions = []

for imp_ds_num in range(1, n_imputed_ds):
    
    imp_ix = imputed_split2_data['.imp'] == imp_ds_num

    y_train_df = imputed_split2_data.loc[imp_ix, ['Molecule.Name'] + target_cols]
    y_val_df = all_training_data.loc[all_training_data['split'] == 'val', ['Molecule.Name'] + target_cols]

    # Scaling training columns 
    scaler = sk.preprocessing.RobustScaler()
    scaler.fit(train_features.loc[train_features['split'] == 'train', cp_cols+chemberta_cols].values)

    X_df = train_features.copy()
    X_df.loc[:, cp_cols+chemberta_cols] = scaler.transform(X_df.loc[:, cp_cols+chemberta_cols].values) 
    X_train_df = X_df.loc[X_df['split']=='train', ['Molecule.Name'] + cp_cols+chemberta_cols]
    X_val_df = X_df.loc[X_df['split']=='val', ['Molecule.Name'] + cp_cols+chemberta_cols]

    X_test_df = test_features.copy()
    X_test_df.loc[:, cp_cols+chemberta_cols] = scaler.transform(X_test_df.loc[:, cp_cols+chemberta_cols].values)
    X_test = X_test_df.loc[:, cp_cols+chemberta_cols].values.astype(np.float32)

    np.testing.assert_array_equal(X_train_df['Molecule.Name'].values, y_train_df['Molecule.Name'].values)
    np.testing.assert_array_equal(X_val_df['Molecule.Name'].values, y_val_df['Molecule.Name'].values)

    # Digitizing target columns
    y_train_properties = digitize_y(y_train_df.loc[:, target_cols], n_cuts=None, bins_by_target=None, remove_nans=False)
    y_val_properties = digitize_y(y_val_df.loc[:, target_cols], n_cuts=None, bins_by_target=y_train_properties, remove_nans=False)

    # Put data in correct format
    y_train_arr = np.concatenate([y_train_properties[target]['values'].reshape(-1, 1) for target in target_cols], axis=1)
    y_val_arr = np.concatenate([y_val_properties[target]['values'].reshape(-1, 1) for target in target_cols], axis=1) 

    X = np.vstack([X_train_df.loc[:, cp_cols+chemberta_cols].values, 
                X_val_df.loc[:, cp_cols+chemberta_cols].values]).astype(np.float32)
    y = np.vstack([y_train_arr, 
                y_val_arr]).astype(np.float32)

    train_ix = np.arange(X_train_df.shape[0])
    val_ix = np.arange(X_train_df.shape[0], X.shape[0])

    # metadata
    n_tasks = y.shape[1]
    n_classes_per_task = [np.unique(y[:, i]).shape[0] for i in range(n_tasks)]
    n_features = X.shape[1]

    # Predictor modules
    backbone = []
    for i in range(backbone_depth):
        backbone.append((f"Backbone_FC_{i}",nn.Linear(n_features, n_features)))
        backbone.append((f"Backbone_ReLU_{i}", nn.ReLU()))
    backbone = nn.Sequential(OrderedDict(backbone))

    head = []
    for i in range(head_depth):
        if i < head_depth - 1: 
            out_dim = n_features
        else:
            out_dim = 1
        head.append((f"Head_FC_{i}",nn.Linear(n_features, out_dim)))
        head.append((f"Head_ReLU_{i}", nn.ReLU()))
    head = nn.Sequential(OrderedDict(head))
        
    # Model
    model = NeuralNet(
        module=OrdinalLogisticMultiTaskModel,
        module__backbone=backbone,
        module__head=head,
        module__n_classes=n_classes_per_task,
        criterion=MultiTaskCumulativeLinkLoss,
        criterion__n_tasks=n_tasks,
        criterion__n_classes_per_task = n_classes_per_task, 
        criterion__loss_reduction = 'inv_num_classes', 
        optimizer=torch.optim.Adam,
        optimizer__weight_decay = weight_decay,
        train_split=lambda ds, y: (torch.utils.data.Subset(ds, train_ix),
                                    torch.utils.data.Subset(ds, val_ix)),
        callbacks=[
            ('ascension', AscensionCallback()),
            ('early_stopping', EarlyStopping(threshold=0.0001, load_best=True,
                                            patience=patience))
        ],
        verbose=0,
        batch_size=train_ix.shape[0],
        max_epochs=1000
    )

    model.fit(X, y)

    # Predict
    mod = model.module_
    mod.eval()

    # Predict ordinal
    y_pred_list = [x.cpu().detach().numpy() for x in mod.forward(torch.as_tensor(X_test))]
    y_preds_ord = np.concatenate([np.argmax(x, axis=1).reshape(-1, 1) for x in y_pred_list], axis=1)

    # Convert to continuous
    y_pred_cont = []
    for i, target in enumerate(target_cols):
        bins = y_train_properties[target]['bins']
        y_pred_cont.append(np.array([bins[x] if not np.isnan(x) else np.nan for x in y_preds_ord[:, i]]).reshape(-1, 1))
    y_pred_cont = np.concatenate(y_pred_cont, axis=1).reshape(X_test.shape[0], n_tasks, 1)

    np.save(f'predictions_imp_{imp_ds_num}.npy', y_pred_cont)

# predicts = np.concatenate(predictions, axis=2)


# In[ ]:


# mod = model.module_
# mod.eval()
# y_pred_list = [x.cpu().detach().numpy() for x in mod.forward(torch.as_tensor(X))]
# y_preds_ord = np.concatenate([np.argmax(x, axis=1).reshape(-1, 1) for x in y_pred_list], axis=1)

# # Convert to continuous
# y_pred_cont = []
# for i, target in enumerate(target_cols):
#     bins = y_train_properties[target]['bins']
#     y_pred_cont.append(np.array([bins[x] if not np.isnan(x) else np.nan for x in y_preds_ord[:, i]]).reshape(-1, 1))
# y_pred_cont = np.concatenate(y_pred_cont, axis=1)

# y_true_train_cont = np.concatenate([y_train_properties[targ]['original'].reshape(-1, 1) for targ in target_cols], axis=1)
# y_true_val_cont = np.concatenate([y_val_properties[targ]['original'].reshape(-1, 1) for targ in target_cols], axis=1)
# y_true_cont = np.concatenate([y_true_train_cont, y_true_val_cont], axis=0)
    
# diff = np.abs(y_pred_cont - y_true_cont)

# train_mask = np.isin(np.arange(diff.shape[0]), train_ix).reshape(-1, 1)
# val_mask = np.isin(np.arange(diff.shape[0]), val_ix).reshape(-1, 1)
# train_mae = np.mean(diff, where=~np.isnan(diff) & train_mask)
# val_mae = np.mean(diff, where=~np.isnan(diff) & val_mask)
# train_mae, val_mae


# In[33]:


