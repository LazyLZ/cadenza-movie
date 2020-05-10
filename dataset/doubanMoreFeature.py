import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.preprocessing import MultiLabelBinarizer
import re
import itertools


def getYear(t):
    s = str(t)
    r = re.match('\d\d\d\d', s)
    return int(r.group()) if r else np.nan


def getCategories(series, map=None):
    if map is not set:
        map = {}
        i = 1
        for t in series.unique():
            if not pd.isna(t):
                map[t] = i
                i += 1
    return series.apply(lambda x: map[x] if not pd.isna(x) else 0)


class DoubanMoreFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, user_map_path, item_map_path, subject_path):
        GENRES_NUM = 3
        TAGS_NUM = 11

        data = pd.read_csv(dataset_path)
        item_map = pd.read_csv(item_map_path)
        user_map = pd.read_csv(user_map_path)
        # read target
        self.targets = self.__preprocess_target(data['rating'].to_numpy()).astype(np.float32)

        # process data
        data = pd.merge(data, user_map, on=['user_id'], how='left')
        data = pd.merge(data, item_map, on=['subject_id'], how='left')

        subjects = pd.read_csv(subject_path)
        subjects = subjects.rename(columns={'id': 'subject_id'})

        # format year
        subjects['year'] = subjects['year'].apply(lambda x: getYear(x))
        # add year category
        subjects.insert(len(subjects.columns), 'year_idx', getCategories(subjects['year']))
        # add genres categories (3 genres)
        genres = subjects['genres'].apply(lambda x: x.split('|') if type(x) == str else [])
        genresMap = set(itertools.chain(*genres))
        for i in range(GENRES_NUM):
            subjects.insert(
                len(subjects.columns),
                'genres%d_idx' % i,
                getCategories(genres.apply(lambda x: x[i] if len(x) > i else None), genresMap)
            )
        data = pd.merge(data, subjects, on='subject_id', how='left')
        print('data', data[['dense_user_id', 'dense_subject_id', 'subject_id']])
        columns = ['dense_user_id', 'dense_subject_id']
        columns.append('year_idx')
        for i in range(GENRES_NUM):
            columns.append('genres%d_idx' % i)
        self.items = data[columns].to_numpy().astype(np.int)
        print('items', self.items)
        # print('ratings', data['rating'].to_numpy())

        print('target', self.targets)
        self.field_dims = np.max(self.items, axis=0) + 1
        # print(self.field_dims)
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        del data
        del subjects
        del item_map
        del user_map

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


if __name__ == '__main__':
    datasetPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\comments.csv'
    item_map_path = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\item_map.csv'
    user_map_path = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\user_map.csv'
    subject_path = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\subjects.csv'

    modelPath = r'C:\Users\laizhi\PycharmProjects\recommender\checkpoints\fm_EPOCH_10_AUC0.8210_.pt'
    itemMapPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\item_map.csv'
    userMapPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\user_map.csv'
    device = torch.device('cuda:0')
    # model = torch.load(modelPath)
    # model = model.to(device)
    #
    # # itemMap = pd.read_csv(itemMapPath)
    # # userMap = pd.read_csv(userMapPath)
    # model.eval()
    # # print('subject count:', len(itemMap))
    # # print('users count:', len(userMap))
    # print(model)
    # print('model load done')
    dataset = DoubanMoreFeatureDataset(
        datasetPath,
        user_map_path,
        item_map_path,
        subject_path
    )
    item, result = dataset[6]
    print('get data', item, result)
    # array = []
    # for i, v in enumerate(item):
    #     num_dims = dataset.field_dims[i]
    #     oneHot = np.zeros([num_dims])
    #     oneHot[v] = 1
    #     print(oneHot, oneHot.shape)
    #     array = np.concatenate((array, oneHot), axis=0)
    # print(array, array.shape)
    # print(item)
    #
    # with torch.no_grad():
    #     fields = torch.tensor(array)
    #     fields = fields.to(device)
    #     y = model(fields)
    #     predicts = y.cpu().tolist()
    #     print('result', y)
