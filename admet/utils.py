
from stepmix.stepmix import StepMix
import seaborn as sns
import numpy as np

def missing_data_clusters(df, ignore=['CXSMILES', 'Molecule Name']): 
    df = df.copy(deep=True)
    for col in df.columns.difference(ignore): 
        df[f'isna-{col}'] = df[col].isna()
    clusters = [3, 4, 5, 6, 7, 8, 9, 10]
    for i in clusters:
        model = StepMix(n_components=i, measurement='binary', verbose=0, random_state=42)
        model.fit(df.filter(regex='^isna'))
        df[f'class_n{i}'] = model.predict(df.filter(regex='^isna'))
    return df

def plot_heatmap(df, n_clusters, ax): 
    ccol = f"class_n{n_clusters}"
    to_plot = df.filter(regex=f'(^isna)|({ccol})').copy()

    to_plot = to_plot.sort_values(by=ccol, ascending=True)
    to_plot[ccol] += 1

    for col in to_plot.columns.difference([ccol]):
        to_plot[col] = to_plot.apply(lambda x: {True: x[ccol],False: np.nan}[x[col]], axis=1)
    
    return sns.heatmap(to_plot, cmap=sns.color_palette('colorblind', n_clusters+1),  square=False, cbar=True, ax=ax)

def transform_targets(df): 
    epsilon = 1e-8
    for col in ['MLM', 'HLM', 'KSOL', 'MDR1-MDCKII']: 
        df.loc[:, f"Log{col}"] = np.log10(np.clip(df[col], a_min=epsilon, a_max=None))
    return df



