import pandas as pd
import random
import math
from numpy import *
import numpy as np
import datetime
from scripts.redindex import reindex

NumOfItems = 0


def GetData(datafile='u.data'):
    '''
    把datafile文件中数据读出来，返回data对象
    :param datafile: 数据源文件名称
    :return: 一个列表，每一个元素是一个元组(userId,movieId)
    '''
    data = []
    try:
        file = open(datafile)
    except:
        print("No such file name" + datafile)
    for line in file:
        line = line.split('\t')
        try:
            data.append((int(line[0]), int(line[1])))
        except:
            pass
    file.close()
    return data


def SplitData(data, M, k, seed):
    '''
    划分训练集和测试集
    :param data:传入的数据
    :param M:测试集占比
    :param k:一个任意的数字，用来随机筛选测试集和训练集
    :param seed:随机数种子，在seed一样的情况下，其产生的随机数不变
    :return:train:训练集 test：测试集，都是字典，key是用户id,value是电影id集合
    '''
    test = dict()
    train = dict()
    random.seed(seed)
    # 在M次实验里面我们需要相同的随机数种子，这样生成的随机序列是相同的
    for user, item in data:
        if random.randint(0, M) != k:
            # 相等的概率是1/M，所以M决定了测试集在所有数据中的比例
            # 选用不同的k就会选定不同的训练集和测试集
            if user not in test.keys():
                test[user] = set()
            test[user].add(item)
        else:
            if user not in train.keys():
                train[user] = set()
            train[user].add(item)
    return train, test


def Recall(train, test, N, k, W, relateditems, k_similar):
    '''
    :param train: 训练集
    :param test: 测试集
    :param N: TopN推荐中N数目
    :param k:
    :return:返回召回率
    '''
    hit = 0  # 预测准确的数目
    totla = 0  # 所有行为总数
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, train, W, relateditems, k, N, k_similar)
        for item in rank:
            if item in tu:
                hit += 1
        totla += len(tu)
    return hit / (totla * 1.0)


def Precision(train, test, N, k, W, relateditems, k_similar):
    '''
    :param train:
    :param test:
    :param N:
    :param k:
    :return:
    '''
    hit = 0
    total = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, train, W, relateditems, k, N, k_similar)
        for item in rank:
            if item in tu:
                hit += 1
        total += N
    return hit / (total * 1.0)


def Coverage(train, test, N, k, W, relateditems, k_similar):
    '''
    计算覆盖率
    :param train:训练集 字典user->items
    :param test: 测试机 字典 user->items
    :param N: topN推荐中N
    :param k:
    :return:覆盖率
    '''
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank = GetRecommendation(user, train, W, relateditems, k, N, k_similar)
        for item in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)


def Popularity(train, test, N, k, W, relateditems, k_similar):
    '''
    计算平均流行度
    :param train:训练集 字典user->items
    :param test: 测试机 字典 user->items
    :param N: topN推荐中N
    :param k:
    :return:覆盖率
    '''
    item_popularity = dict()
    for user, items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, train, W, relateditems, k, N, k_similar)
        for item in rank:
            if item != 0:
                ret += math.log(1 + item_popularity[item])
                n += 1
    ret /= n * 1.0
    return ret


def getW(train):
    # train本身已经是用户->物品倒排表
    # W[u][v]表示物品u和物品v的相似度
    W = zeros([NumOfItems, NumOfItems], dtype=float16)
    # C[u][v]表示喜欢u有喜欢v物品的用户有多少个
    C = zeros([NumOfItems, NumOfItems], dtype=float16)
    # N[u]表示有多少用户喜欢物品u
    N = zeros([NumOfItems], dtype=float16)

    item_relateditems = dict()

    for user, items in train.items():
        for item1 in items:
            N[item1] += 1
            for item2 in items:
                if item1 == item2:
                    continue
                if item1 not in item_relateditems:
                    item_relateditems[item1] = set()
                item_relateditems[item1].add(item2)
                # 使用IUF矫正的ItemCF
                C[item1][item2] += (1 / math.log(1 + len(items) * 1.0))

    for item1 in range(0, NumOfItems):
        if item1 in item_relateditems:
            for item2 in item_relateditems[item1]:
                W[item1][item2] = C[item1][item2] / sqrt(N[item1] * N[item2])

    return W, item_relateditems


def k_similar_item(W, item_relateditems, k):
    '''
    :param W:
    :param item_relateditems:
    :param k:
    :return:返回一个字典，key是每个item，value是item对应的k个最相似的物品
    '''
    begin = datetime.datetime.now()

    k_similar = dict()
    for i in range(0, NumOfItems):
        relateditems = dict()
        try:
            for x in item_relateditems[i]:
                relateditems[x] = W[i][x]
            relateditems = sorted(relateditems.items(), key=lambda x: x[1], reverse=True)
            k_similar[i] = set(dict(relateditems[0:k]))  # 返回k个与物品i最相似的物品
        except KeyError:
            # print(i, " doesn't have any relateditems")
            k_similar[i] = set()
            for x in range(0, k + 1):
                k_similar[i].add(x)
    end = datetime.datetime.now()
    print("it takes ", (end - begin).seconds, " seconds to get k_similar_item for all items.")
    return k_similar


def GetRecommendation(user, train, W, relateditems, k, N, k_similar_items):
    '''
    :param user: 目标用户
    :param train: 训练集 字典user->items
    :param W: 物品相似度矩阵
    :param relateditems: 字典 items->相关item
    :param k: 从目标用户历史兴趣列表中选取k个与推荐item最为相似的物品
    :param N: 给目标用户推荐N个物品
    :param k_similar_items: 一个字典，key是每个item，value是item对应的k个最相似的物品
    :return:
    '''
    rank = dict()  # key是电影id，value是兴趣大小

    for i in range(NumOfItems):
        rank[i] = 0

    possible_recommend = set()
    for item in train[user]:
        ##返回训练集中和目标用户历史兴趣物品相似度不为0的物品item
        possible_recommend = possible_recommend.union(relateditems[item])

    for item in possible_recommend:
        k_items = k_similar_items[item]  # 返回与item最为相似的k个物品
        for i in k_items:
            if i in train[user]:  # 且返回的k个物品必须在目标用户历史兴趣物品列表里
                rank[item] += 1.0 * W[item][i]

    ##rank字典，key是itemId，value是用户user对这个推荐的itemId的兴趣程度，前提是这个item不能出现在用户user历史兴趣物品列表里
    for rank_key in rank:
        if rank_key in train[user]:  ##如果推荐的item出现在用户历史兴趣物品列表里，则赋值0
            rank[rank_key] = 0
    # 按照用户user对推荐的item兴趣程度，从大到小排序，推荐N个物品
    return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:N])


def evaluate(train, test, N, k):
    ##计算一系列评测标准

    recommends = dict()

    W, relateditems = getW(train)
    k_similar = k_similar_item(W, relateditems, k)
    for user in test:
        recommends[user] = GetRecommendation(user, train, W, relateditems, k, N, k_similar)

    recall = Recall(train, test, N, k, W, relateditems, k_similar)
    precision = Precision(train, test, N, k, W, relateditems, k_similar)
    coverage = Coverage(train, test, N, k, W, relateditems, k_similar)
    popularity = Popularity(train, test, N, k, W, relateditems, k_similar)
    return recall, precision, coverage, popularity


def test2():
    N = int(input("input the number of recommendations: \n"))
    k = int(input("input the number of related items: \n"))
    data = GetData()
    train, test = SplitData(data, 2, 1, 1)
    del data
    recall, precision, coverage, popularity = evaluate(train, test, N, k)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Coverage: ", coverage)
    print("Popularity: ", popularity)


if __name__ == '__main__':
    subjects = pd.read_csv(r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\subjects.csv')
    ratings = pd.read_csv(r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\comments.csv')
    itemMap = pd.read_csv(r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\item_map.csv')
    userMap = pd.read_csv(r'C:\Users\laizhi\PycharmProjects\recommender\data\douban18m\user_map.csv')
    print('read file done')
    NumOfItems = len(subjects)
    print(NumOfItems)

    ratings = pd.merge(ratings, userMap, on=['user_id'], how='left')
    ratings = pd.merge(ratings, itemMap, on=['subject_id'], how='left')
    ratings = ratings[['dense_user_id', 'dense_subject_id', 'rating', 'create_time']]

    positiveRatings = ratings[ratings['rating'] >= 3]
    positiveRatings = positiveRatings.reindex(np.random.permutation(positiveRatings.index))
    positiveRatings = positiveRatings.iloc[:1000000]
    del ratings
    del itemMap
    del userMap
    # split
    rate = 0.1
    total = len(positiveRatings)
    testData = positiveRatings.iloc[0:int(rate * total)]
    trainData = positiveRatings.iloc[int(rate * total):]

    trainGroup = trainData.groupby(['dense_user_id'])
    testGroup = testData.groupby(['dense_user_id'])

    test = dict()
    train = dict()
    seed = 1232
    random.seed(seed)
    for key, df in trainGroup:
        # print(key, df['dense_subject_id'].tolist())
        train[key] = set(df['dense_subject_id'].tolist())

    for key, df in testGroup:
        # print(key, group)
        test[key] = set(df['dense_subject_id'].tolist())
    # return train, test

    N = 50
    k = 10
    recall, precision, coverage, popularity = evaluate(train, test, N, k)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Coverage: ", coverage)
    print("Popularity: ", popularity)
