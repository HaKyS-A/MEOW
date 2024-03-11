import json
import os

import torch

from dataset.JsonToDataset import CreateDataset
from Train_OneRound import trainIt
from utils.util import set_random_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We consider the spy identification as a four-class classification problem: A as the undercover / B as the undercover / C as the undercover / D as the undercover.


def train_Tworound(config_TwoRound):
    '''
    train expert models (GAT_TwoRound) in the dataset of the games with two rounds

    Args:
    config_TwoRound: the configuration of the training process
    '''
    _, datasetTwice = CreateDataset(config_TwoRound['dataset_path'])
    print('------------------------------------------------------------')
    model_save = config_TwoRound['model_save'][:-1]
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    for i in range(config_TwoRound['original_random_state'], 11):
        print(f'Training the second expert model with random_state={i}')
        trainIt(
            dataset=datasetTwice,
            model_name='TwoRound',
            random_state=i,
            config=config_TwoRound,
            mode=1
        )
    print('Training complete')


if __name__ == '__main__':
    with open('config-TwoRound.json', 'r', encoding='utf-8') as file:
        config_TwoRound = json.load(file)
    for i in config_TwoRound:
        print(f'{i}:{config_TwoRound[i]}')
    set_random_seed(config_TwoRound['seed'])
    train_Tworound(config_TwoRound)
