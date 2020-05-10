from model import *
from dataset import *
import torch
from torch.utils.data import DataLoader
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import os


def get_dataset(name, path):
    if name == 'movielens':
        return MovieLensDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    elif name == 'douban':
        return DoubanDataset(
            path,
            item_map_path=r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\item_map.csv',
            user_map_path=r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\user_map.csv'
        )
    elif name == 'doubanMore':
        return DoubanMoreFeatureDataset(
            path,
            item_map_path=r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\item_map.csv',
            user_map_path=r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\user_map.csv',
            subject_path=r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\subjects.csv'
        )
    else:
        raise ValueError('unknown dataset name: ' + name)


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    print('field_dims', field_dims)
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(
            field_dims=field_dims,
            embed_dim=16
        )
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(
            field_dims=field_dims,
            embed_dim=4
        )
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(
            field_dims=field_dims,
            embed_dim=16,
            mlp_dims=(16, 16),
            dropout=0.2
        )
    elif name == 'wd':
        return WideAndDeepModel(
            field_dims=field_dims,
            embed_dim=16,
            mlp_dims=(512, 256, 128),
            dropout=0.5
        )
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(
            field_dims=field_dims,
            embed_dim=16,
            mlp_dims=(16,),
            method='inner',
            dropout=0.2
        )
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(
            field_dims,
            embed_dim=16,
            num_layers=3,
            mlp_dims=(128, 64),
            dropout=0.5
        )
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLensDataset) or isinstance(dataset, DoubanDataset)
        return NeuralCollaborativeFiltering(
            field_dims, embed_dim=16,
            mlp_dims=(64, 32, 16),
            dropout=0.2,
            user_field_idx=dataset.user_field_idx,
            item_field_idx=dataset.item_field_idx
        )
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(
            field_dims,
            embed_dim=16,
            mlp_dims=(256, 256, 256),
            dropout=0.5
        )
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400),
            dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, data_loader, criterion, device, log_interval=1000):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        fields, target = fields.to(device), target.to(device)
        # print('train input', fields)
        model.zero_grad()
        y = model(fields)
        loss = criterion(y, target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0
    return total_loss


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    print('Model: ', model_name)
    writer = SummaryWriter(r'C:\Users\laizhi\PycharmProjects\recommender\logs')
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    model = get_model(model_name, dataset).to(device)
    print('model', model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch_i in range(epoch):
        total_loss = train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        writer.add_scalar('model/loss', total_loss, epoch_i)
        writer.add_scalar('performance/AUC', auc, epoch_i)
        torch.save(model, '%s/%s_%s_EPOCH_%d_AUC%.4f_.pt' % (save_dir, dataset_name, model_name, epoch_i, auc))
    auc = test(model, test_data_loader, device)

    print('scripts auc:', auc)
    torch.save(model, '%s/%s_%s_EPOCH_%d_AUC%.4f_.pt' % (save_dir, dataset_name, model_name, epoch, auc))


if __name__ == '__main__':
    # config = {
    #     'dataset_name': 'movie',
    #     'dataset_path': '',
    #     'model_name': 'fm',
    #     'epoch': 15,
    #     'learning_rate': 1e-3,
    #     'batch_size': 2048,
    #     'weight_decay': 1e-6,
    #     'device': 'cuda:0',
    #     'save_dir': 'checkpoints'
    # }
    main(
        dataset_name='doubanMore',
        dataset_path=r'C:\Users\laizhi\PycharmProjects\recommender\data\douban10m\comments.csv',
        # dataset_path=r'C:\Users\laizhi\PycharmProjects\recommender\data\movieLens1m\ratings.csv',
        # dataset_name='movielens',
        # dataset_path='C:\\Users\\laizhi\\PycharmProjects\\recommender\\data\\movieLens1m\\ratings.csv',
        model_name='dfm',
        epoch=100,
        learning_rate=1e-3,
        batch_size=2048,
        weight_decay=2e-5,
        device='cuda:0',
        save_dir='../checkpoints'
    )
