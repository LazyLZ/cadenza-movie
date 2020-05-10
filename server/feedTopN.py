from dataset import *
from scripts.train import get_dataset, get_model
import torch
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import time
import numpy as np
import pymongo
from model.gmf import GeneralizedMatrixFactorizationModel
from scripts.redindex import reindex
dbUrl = 'mongodb://localhost:27017'
dbName = 'douban'
datasetName = 'douban'
ratingPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\comments.csv'
modelPath = r'C:\Users\laizhi\PycharmProjects\recommender\checkpoints\douban_ncf_EPOCH_4_AUC0.8165_.pt'
# subjectPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\subjects.csv'
itemMapPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\item_map.csv'
userMapPath = r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\user_map.csv'

client = pymongo.MongoClient(dbUrl)
db = client[dbName]

# dataset = get_dataset(
#     datasetName,
#     ratingPath,
# )
# model = get_model(datasetName, dataset)
device = torch.device('cuda:0')

# model = torch.load(modelPath)
model = GeneralizedMatrixFactorizationModel(
    num_users=40535,
    num_items=74515,
    latent_dim=8
)
model.load_state_dict(torch.load(
    r'C:\Users\laizhi\PycharmProjects\neural-collaborative-filtering\src\checkpoints\gmf_factor8neg4-implict_Epoch32_HR0.4383_NDCG0.2499.model'
))

model = model.to(device)
# data_loader = DataLoader(
#     dataset,
#     batch_size=2048,
#     num_workers=8
# )


# 取评论最多的用户
ratings = pd.read_csv(ratingPath)
group = ratings.groupby('user_id')
counts = group.size()
counts = pd.DataFrame({'user_id': counts.index.values, 'count': counts})
counts.index.name = None
counts = counts[['user_id', 'count']]
counts.reset_index()
userSet = set(counts[counts['count']>80]['user_id'].values)
ratings = ratings[ratings['user_id'].isin(userSet)]
print('ratings', ratings)


userMap = reindex(ratings, 'user_id', 'dense_user_id')
itemMap = reindex(ratings, 'subject_id', 'dense_subject_id')

del ratings

# itemMap = pd.read_csv(itemMapPath)
# userMap = pd.read_csv(userMapPath)

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
        'dense_subject_id': itemMap['dense_subject_id']
    })

    with torch.no_grad():
        start = time.time()
        # fields = torch.tensor(metrics.values.tolist())
        # fields = fields.to(device)
        # y = model(fields)

        users = metrics['dense_user_id'].values.tolist()
        items = metrics['dense_subject_id'].values.tolist()
        users = torch.tensor(users, device=device)
        items = torch.tensor(items, device=device)
        y = model(users, items)


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
    print(histories)

    df = pd.merge(metrics, histories, how='left')
    # print(df)
    recommends = df[df['rating'].isna()]
    recommends = recommends.sort_values(by='score', ascending=False)
    # print(df)
    print(recommends)

    topN = recommends.iloc[:N]['dense_subject_id']

    # realItemId = itemMap[itemMap['dense_subject_id'].isin(topN)]
    topN = pd.merge(topN, itemMap, how='left')
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
    # 1404088
    list, histories = getTopN('32585937', 10)
    print('recommend list')
    for id in list:
        print(db['subject'].find_one({'id': str(id)}, {"_id": 0, "title": 1, "ratings_count": 1}))
    print()
    print('history list')
    for id in histories:
        print(db['subject'].find_one({'id': str(id)}, {"_id": 0, "title": 1}))
