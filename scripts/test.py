from dataset import *
from scripts.train import get_dataset, get_model
import torch
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import numpy as np

datasetName = 'movielens'
datasetPath = '../data/movieLens1m/ratings.csv'
dataset = get_dataset(datasetName, datasetPath)
# model = get_model(datasetName, dataset)
device = torch.device('cuda:0')
modelPath = r'C:\Users\laizhi\PycharmProjects\recommender\checkpoints\movielens_fm_EPOCH_15_AUC0.8038_.pt'
model = torch.load(modelPath)
model = model.to(device)
data_loader = DataLoader(
    dataset,
    batch_size=2048,
    num_workers=8
)

print(model)
if __name__ == '__main__':
    # because start with 1
    movies = pd.read_csv('../data/movieLens1m/movies.csv')
    ratings = pd.read_csv('../data/movieLens1m/ratings.csv')
    users = pd.read_csv('../data/movieLens1m/users.csv')

    u = users.iloc[np.random.randint(len(users))]
    print('recommend for user:')
    print(u)
    uid = u['origin_uid']
    metrics = pd.DataFrame({
        'origin_uid': uid,
        'origin_iid': movies['origin_iid']
    })
    model.eval()
    with torch.no_grad():
        fields = torch.tensor(metrics.values.tolist())
        fields = fields.to(device)
        y = model(fields)
        predicts = y.cpu().tolist()
        metrics.insert(2, 'score', predicts)
    metrics = metrics.sort_values(by='score', ascending=False)

    histories = ratings[ratings['origin_uid'] == uid]

    # print(histories)
    # print(metrics)
    df = pd.merge(metrics, histories, how='left')
    # print(df)
    recommends = df[df['rating'].isna()]
    recommends = recommends.sort_values(by='score', ascending=False)
    # print(df)
    # print(recommends)
    K = 10
    topK = recommends.iloc[:K]['origin_iid']
    print(topK)
    topK = pd.merge(
        movies,
        pd.DataFrame({
            'origin_iid': topK,
            'rank': np.arange(K)
        }),
        how='left'
    )
    topK = topK[topK['rank'].notna()]
    topK = topK.sort_values(by='rank')
    print(topK)
    print('target user id:', uid)
    print('recommend movies:')
    print(topK['title'])

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
