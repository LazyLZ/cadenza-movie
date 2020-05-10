# Reindex
import pandas as pd
import numpy as np


def reindex(data, colName, newName, saveName=''):
    col = data[[colName]].drop_duplicates().reindex()
    col[newName] = np.arange(len(col))
    if saveName:
        print('save', saveName)
        col.to_csv(saveName, index=False)
    return col


if __name__ == '__main__':
    data = pd.read_csv(r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\comments.csv')
    reindex(data, 'user_id', 'dense_user_id', './user_map.csv')
    # reindex(data, 'subject_id', 'dense_subject_id', './item_map.csv')