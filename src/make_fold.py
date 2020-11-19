from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
import pandas as pd
import numpy as np

def make_fold(df, num_fold=5, how='kfold'):
    df['fold'] = -1
    if how == 'kfold':
        kf = KFold(n_splits=num_fold, random_state=42, shuffle=True)
    # elif how == 'group':
    #     kf = GroupKFold(n_splits=num_fold, random_state=42)
    elif how == 'stratified':
        kf = StratifiedKFold(n_splits=num_fold, random_state=42, shuffle=True)

    for fold, (_, val_idx) in enumerate(kf.split(df['id'], df['landmark_id'])):
        df.iloc[val_idx, -1] = fold

    return df
