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

# 取评论最多的用户
ratings = pd.read_csv(ratingPath)
group = ratings.groupby('user_id')
counts = group.size()
counts = pd.DataFrame({'user_id': counts.index.values, 'count': counts})
counts.index.name = None
counts = counts[['user_id', 'count']]
counts.reset_index()
userSet = set(counts[counts['count'] > 80]['user_id'].values)
ratings = ratings[ratings['user_id'].isin(userSet)]
print('ratings', ratings)


userMap = reindex(ratings, 'user_id', 'dense_user_id')
itemMap = reindex(ratings, 'subject_id', 'dense_subject_id')

num_users = len(userMap)
num_items = len(itemMap)


counts = ratings.groupby('subject_id')
counts = counts.size()
counts = counts[counts > 400]
# print(counts)
counts = counts.index.values
counts = pd.DataFrame({'subject_id': counts})
# print('counts', counts)

itemMap = pd.merge(counts, itemMap, how='left')
# print(itemMap)

del ratings
del counts


# model = torch.load(modelPath)
model = GeneralizedMatrixFactorizationModel(
    num_users=num_users,
    num_items=num_items,
    latent_dim=8
)
model.load_state_dict(torch.load(
    r'C:\Users\laizhi\PycharmProjects\neural-collaborative-filtering\src\checkpoints\gmf_factor8neg4-implict_Epoch32_HR0.4383_NDCG0.2499.model'
))

model = model.to(device)


# itemMap = pd.read_csv(itemMapPath)
# userMap = pd.read_csv(userMapPath)

# 应该从数据库读
# ratings = pd.read_csv(ratingPath)
model.eval()
print('subject count:', len(itemMap))
print('users count:', len(userMap))
print(model)
print('model load done')


def getDist(A, B):
    # print(A, B)
    num = sum(A * B)
    # print('num', num)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    # print('denom', np.linalg.norm(A), np.linalg.norm(B))
    return num / denom


class EmbeddingNet(torch.nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=8)

    def forward(self, x):
        return self.embedding(x)


if __name__ == '__main__':
    # getTopN('25643521', 10)
    key = list(model.state_dict().keys())[1]
    params = model.state_dict()[key]
    embeddingNet = EmbeddingNet(num_embeddings=num_items)
    # embeddingNet = FeaturesEmbedding([len(userMap), len(itemMap)], 16)
    embeddingNet = embeddingNet.to(device)
    embeddingNet.load_state_dict({'embedding.weight': params})

    metrics = pd.DataFrame({
        # 'dense_user_id': 88,
        'dense_subject_id': itemMap['dense_subject_id']
    })
    print(embeddingNet)
    with torch.no_grad():
        fields = torch.tensor(metrics['dense_subject_id'].values.T).long()
        fields = fields.view([1, len(fields)])
        fields = fields.transpose(0, 1)
        print(fields.shape, fields)
        fields = fields.to(device)
        start = time.time()
        y = embeddingNet(fields)
        y = y.squeeze()
        predicts = y.cpu().tolist()
        print(y.shape, y)
        metrics.insert(1, 'embedding', predicts)
    # 26930504 烟花
    # 1291841 教父
    # 3011235 哈利波特与死亡圣器下
    sid = 3011235
    print('item example:', db['subject'].find_one({'id': str(sid)}, {'_id': 0, 'id': 1, 'title': 1}))
    denseItemId = itemMap[itemMap['subject_id'] == int(sid)]
    if len(denseItemId):
        denseItemId = denseItemId.iloc[0]['dense_subject_id']
    else:
        print('item not found')
        exit()

    print(denseItemId)
    list = []
    embeddingList = metrics['embedding'].to_list()
    target = metrics[metrics['dense_subject_id'] == denseItemId]
    target = target.iloc[0]
    for e in embeddingList:
        A = np.array(e)
        B = np.array(target['embedding'])
        # print('E', A.shape)
        # print('T', B.shape)
        list.append(getDist(A, B))
    # print(list)
    metrics.insert(2, 'distance', list)
    metrics = metrics.sort_values(by='distance', ascending=False)
    print(metrics)
    df = pd.merge(metrics, itemMap, how='left')
    # df = df.drop(columns='dense_user_id')
    for i in range(50):
        item = df.iloc[i]
        id = item['subject_id']
        print('similar', item['distance'], db['subject'].find_one({'id': str(id)}, {'_id': 0, 'id': 1, 'title': 1}))
