import numpy as np
import pandas as pd
import torch.utils.data


class DoubanDataset(torch.utils.data.Dataset):
    """
    Douban Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: Douban dataset path (comments.csv)

    """

    def __init__(self, dataset_path, user_map_path, item_map_path, sep=',', engine='c', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header)
        item_map = pd.read_csv(item_map_path, sep=sep, engine=engine, header=header)
        user_map = pd.read_csv(user_map_path, sep=sep, engine=engine, header=header)
        data = pd.merge(data, user_map, on=['user_id'], how='left')
        data = pd.merge(data, item_map, on=['subject_id'], how='left')
        data = data[['dense_user_id', 'dense_subject_id', 'rating', 'create_time']]
        # print(data)
        data = data.to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int)
        # print(self.items)
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        # print(self.field_dims)
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


if __name__ == '__main__':
    datasetPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\comments.csv'
    item_map_path = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\item_map.csv'
    user_map_path = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\user_map.csv'
    modelPath = r'C:\Users\laizhi\PycharmProjects\recommender\checkpoints\fm_EPOCH_10_AUC0.8210_.pt'
    # subjectPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\subjects.csv'
    itemMapPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\item_map.csv'
    userMapPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\user_map.csv'
    device = torch.device('cuda:0')
    model = torch.load(modelPath)
    model = model.to(device)

    # itemMap = pd.read_csv(itemMapPath)
    # userMap = pd.read_csv(userMapPath)
    model.eval()
    # print('subject count:', len(itemMap))
    # print('users count:', len(userMap))
    print(model)
    print('model load done')
    dataset = DoubanDataset(
        datasetPath,
        user_map_path,
        item_map_path
    )
    item, result = dataset[9999]
    print('get data', item, result)
    array = []
    for i, v in enumerate(item):
        num_dims = dataset.field_dims[i]
        oneHot = np.zeros([num_dims])
        oneHot[v] = 1
        print(oneHot, oneHot.shape)
        array = np.concatenate((array, oneHot), axis=0)
    print(array, array.shape)
    print(item)

    with torch.no_grad():
        fields = torch.tensor(array)
        fields = fields.to(device)
        y = model(fields)
        predicts = y.cpu().tolist()
        print('result', y)

