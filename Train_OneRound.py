import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.loader import DataLoader

from dataset.JsonToDataset import CreateDataset
from model.model_OneRound import GAT_OneRoud
from model.model_TwoRound import GAT_TwoRoud
from utils.evaluate import KFold_acc, cal_acc_loss
from utils.util import set_random_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We consider the spy identification as a four-class classification problem: A as the undercover / B as the undercover / C as the undercover / D as the undercover.


def train_epoch(model, train_loader, optimizer, criterion):
    '''
    Training completed for each epoch

    Args:
    model: the expert model
    train_loader: the DataLoader of dataset
    optimizer: determine how the model's weights are updated during training
    criterion: loss function
    '''
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_num = 0

    for data in train_loader:
        optimizer.zero_grad()
        output = model(data.x_dict, data.edge_index_dict)
        y0 = data['players'].y.float()
        y0 = torch.reshape(y0, (-1, 4))
        y0 = torch.argmax(y0, dim=1)
        loss = criterion(output, y0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # calculate accuracy
        result = torch.argmax(output, dim=1)
        # tensor(batchsize)
        length = len(result)
        total_num += length
        for i in range(length):
            if result[i] == y0[i]:
                total_acc += 1.0

    return total_loss / len(train_loader), total_acc / total_num


def trainIt(dataset, model_name, random_state, config, mode):
    '''
    Implement training for an expert model

    Args:
    datase: the total list of HeteroData
    model_name: in our experiment, model_name='OneRound' or model_name='TwoRound', and it will be used in the name of saved model.
    random_state：parameter value used in the sklearn function for splitting datasets.
    config: a configuration for model training
    mode: mode=0 means the first expert model(GAT_OneRound), mode=1 means the second expert model (GAT_TwoRound)
    '''
    assert model_name in ['OneRound', 'TwoRound'], 'The model_name should in [One_Round, Two_Round]'
    assert mode in [0, 1], 'mode should in [0,1]'

    trainval_dataset, test_dataset = train_test_split(dataset, train_size=config['train_size'], random_state=random_state)  #（6:2）:2,（training:validation):test
    print(f'The length of training and validation set is {len(trainval_dataset)}，the length of test set is {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)

    # 4-fold cross-validation
    kf = KFold(n_splits=config['folds'])
    kf.get_n_splits(trainval_dataset)
    model_best_record = []  # save the 4 best models in 4 folds
    fold = 0
    for train_index, val_index in kf.split(trainval_dataset):
        train_dataset = [trainval_dataset[i] for i in train_index]
        val_dataset = [trainval_dataset[i] for i in val_index]
        # Load the data into a DataLoader
        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False)
        # Define the model, loss function, and optimizer. Train from scratch for each fold
        if mode == 0:
            model = GAT_OneRoud(
                hidden_channels=config['hidden_channels'],
                hidden_channels1=config['hidden_channels1'],
                heads=config['heads'],
                out_channels=4,
                aggr_way=config['aggr'],
                dropvalue=config['p'],
            ).to(device)
        else:
            model = GAT_TwoRoud(
                hidden_channels=config['hidden_channels'],
                hidden_channels1=config['hidden_channels1'],
                heads=config['heads'],
                out_channels=4,
                aggr_way=config['aggr'],
                dropvalue=config['p'],
            ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        model_best = model
        lossmin = 999
        epoch = 0
        curr_patience = 0
        print(f'Training Fold{fold}...')
        while True:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f},Train Acc:{train_acc:.4f}')
            epoch = epoch + 1
            val_loss, val_acc = cal_acc_loss(model, val_loader, criterion)
            print(f'Val Loss:{val_loss:.4f},Val Acc:{val_acc:.4f}')
            if lossmin > val_loss and epoch >= config['patience']:
                lossmin = val_loss
                curr_patience = 0
                model_best = model
            else:
                curr_patience += 1
            if curr_patience >= config['patience']:
                print(f'Fold{fold}...,Early stop in {epoch},')
                break
            if epoch >= 500:  # up to 500 epochs
                print(f'Fold{fold}...,Early stop in {epoch},')
                break

        # Save and test the model
        real_save = f'{config["model_save"]}{model_name}-random-{random_state}-bestmodel-{fold}.pth'
        model_best_record.append(model_best)
        print(f'save the best model in {real_save}')
        torch.save(model_best, real_save)

        print(f'Fold{fold},over')
        fold += 1

    print('Four-fold cross-validation completed')
    trainval_loader = DataLoader(trainval_dataset, batch_size=config['train_batch_size'], shuffle=False)
    trainval_acc = KFold_acc(model_best_record, trainval_loader)
    test_acc = KFold_acc(model_best_record, test_loader)
    print(f'trainval_acc_avg={trainval_acc},test_acc_avg={test_acc}')


def train_Oneround(config_OneRound):
    '''
    train expert models (GAT_OneRound) in the dataset of the first round of each game

    Args:
    config_OneRound: the configuration of the training process
    '''
    datasetOnce, _ = CreateDataset(config_OneRound['dataset_path'])
    print('------------------------------------------------------------')
    model_save = config_OneRound['model_save'][:-1]
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    # ten different train-test sets
    for i in range(config_OneRound['original_random_state'], 11):
        print(f'Training the first expert model with random_state={i}')
        trainIt(
            dataset=datasetOnce,
            model_name='OneRound',
            random_state=i,
            config=config_OneRound,
            mode=0
        )
    print('Training complete')


if __name__ == '__main__':
    with open('config-OneRound.json', 'r', encoding='utf-8') as file:
        config_OneRound = json.load(file)
    for i in config_OneRound:
        print(f'{i}:{config_OneRound[i]}')
    set_random_seed(config_OneRound['seed'])
    train_Oneround(config_OneRound)
