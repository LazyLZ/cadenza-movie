from dataset import *
from scripts.train import get_dataset, get_model
import torch
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import time
import numpy as np
import pymongo

from scripts.redindex import reindex

dbUrl = 'mongodb://localhost:27017'
dbName = 'douban'
datasetName = 'douban'
ratingPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\comments.csv'
modelPath = r'C:\Users\laizhi\PycharmProjects\recommender\checkpoints\doubanMore_fm_EPOCH_5_AUC0.8304_.pt'
subjectPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\subjects.csv'
itemMapPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\item_map.csv'
userMapPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\user_map.csv'

client = pymongo.MongoClient(dbUrl)
db = client[dbName]

# dataset = get_dataset(
#     datasetName,
#     ratingPath,
# )
# model = get_model(datasetName, dataset)
device = torch.device('cuda:0')

model = torch.load(modelPath)
model = model.to(device)

itemMap = pd.read_csv(itemMapPath)
userMap = pd.read_csv(userMapPath)

subjects = pd.read_csv(subjectPath)
# print(data)
subjects = subjects[['id', 'genres']]
subjects = subjects.rename(columns={'id': 'subject_id'})
subjects['genres'] = subjects['genres'].apply(lambda x: x.split('|')[0] if type(x) == str else None)
genresList = subjects['genres'].unique()
genresMap = {}
for i, g in enumerate(genresList):
    genresMap[g] = i + 1
subjects.insert(2, 'genres_index', subjects['genres'].apply(lambda x: genresMap[x] if x is not None else 0))

itemMap = pd.merge(itemMap, subjects, on=['subject_id'], how='left')

# 应该从数据库读
# ratings = pd.read_csv(ratingPath)
model.eval()
print('subject count:', len(itemMap))
print('users count:', len(userMap))
print(model)
print('model load done')


def getTopN(userId, N):
    # u = users.iloc[np.random.randint(len(users))]
    # print('recommend for user:')
    # print(u)
    # uid = u['origin_uid']
    denseUserId = userMap[userMap['user_id'] == int(userId)]
    if len(denseUserId):
        denseUserId = denseUserId.iloc[0]['dense_user_id']
    else:
        print('user not found')
        return []
    print('denseUserId', denseUserId)

    metrics = pd.DataFrame({
        'dense_user_id': denseUserId,
        'dense_subject_id': itemMap['dense_subject_id'],
        'genres_index': itemMap['genres_index']
    })
    print('metrics input', metrics)

    with torch.no_grad():
        start = time.time()
        fields = torch.tensor(metrics.values.tolist())
        fields = fields.to(device)
        y = model(fields)

        predicts = y.cpu().tolist()
        end = time.time()
        metrics.insert(2, 'score', predicts)
    # metrics = metrics.sort_values(by='score', ascending=False)

    print('Model computed use', end - start, 'seconds')
    list = {'subject_id': [], 'rating': []}
    for doc in db['comment'].find({'author.id': userId}, {"_id": 0}):
        list['subject_id'].append(int(doc['subject_id']))
        list['rating'].append(int(doc['rating']['value']))
    # print(list)
    # histories = ratings[ratings['user_id'] == int(userId)]
    histories = pd.DataFrame(list)
    histories = pd.merge(histories, itemMap, how='left')
    print('histories', histories)
    print('metrics', metrics)

    df = pd.merge(metrics, histories, how='left')
    # print(df)
    recommends = df[df['rating'].isna()]
    recommends = recommends.sort_values(by='score', ascending=False)
    # print(df)
    print(recommends)

    topN = recommends.iloc[:N]['dense_subject_id']

    # realItemId = itemMap[itemMap['dense_subject_id'].isin(topN)]
    topN = pd.merge(topN, itemMap,on=['dense_subject_id'], how='left')
    print(topN)
    return topN['subject_id'].values.tolist(), histories['subject_id'].values.tolist()
    # topN = pd.merge(
    #     movies,
    #     pd.DataFrame({
    #         'origin_iid': topN,
    #         'rank': np.arange(N)
    #     }),
    #     how='left'
    # )
    # topN = topN[topN['rank'].notna()]
    # topN = topN.sort_values(by='rank')
    # print(topN)
    # print('movies:')
    # print(topN['title'])
    # return topN['origin_iid'].values.tolist()


# print(model)
# if __name__ == '__main__':
# for fields, targets in data_loader:
#     print(fields)
#     print(targets)
#     break
# for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
#     fields, target = fields.to(device), target.to(device)
#     y = model(fields)
#     targets.extend(target.tolist())
#     predicts.extend(y.tolist())

# print(targets)
# print(predicts)

if __name__ == '__main__':
    # 32585937 二次元
    # 1005928
    # 1404088 韩剧
    list, histories = getTopN('32585937', 50)
    print('recommend list')
    for id in list:
        print(db['subject'].find_one({'id': str(id)}, {"_id": 0, "title": 1, "ratings_count": 1}))
    print()
    print('history list')
    for id in histories:
        print(db['subject'].find_one({'id': str(id)}, {"_id": 0, "title": 1}))
