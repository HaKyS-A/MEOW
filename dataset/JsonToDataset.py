import json
import pickle

import torch
from torch_geometric.data import HeteroData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def json_to_pyg_graph(json_tendency, json_vote, json_player):
    '''
    Construct a heterogeneous graph based on game data

    Args:
    json_tendency: [{'player1':{'for':[...],'against':[...],'no comment':[...]}}]. The tendencies of players extracted from the JSON file, ensure that the corresponding content for 'for' and 'against' is represented using lists.
    json_vote：[{'player1':'player_i','player2':'...',...}]. The votes of players ectracted from the JSON file.
    json_player: The basic information of players,[['player1',...,'player4'],['player1',description word,the log file,the identify(0/1)],...]

    Return：
    the first element is a list, comprising of the `HeteroData` of the first-round of each game (3 edge types,['players', 'for'/'against'/'vote', 'players'])
    the second element is a list, comprising of the `HeteroData` of the game with two rounds (if the game only has one round, return a empty list) (6 edge types,['players', 'for'/'against'/'vote'/'for1'/'against1'/'vote1', 'players'])
    the third element is a mapping table between players' names and corresponding IDs in a game
    '''
    hetero_data_once = []
    hetero_data_twice = []
    xlabel = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # shwo that the player has no for/against/vote tendency
    noidea = ['', '无', '无法选择', '不投票']
    LabelInf = {}  # LabelInf[player]=identity,eq:LabelInf['Alice']=0[folk]/1[spy]
    personId = {}  # personId[player]=index_id
    # Read character information
    for i in range(len(json_player[0])):
        LabelInf[json_player[i + 1][0]] = json_player[i + 1][3]
        personId[json_player[i + 1][0]] = i
    i = 0  # record the current round
    # the infomation of each round
    for round_data, vote_data in zip(json_tendency, json_vote):
        # Constructe edge information (for or against)
        edge_for_index = []
        edge_against_index = []
        edge_vote_index = []
        for person, opinions in round_data.items():
            for supporter in opinions['for']:
                if supporter in personId and supporter in noidea:
                    continue
                edge_for_index.append((personId[person], personId[supporter]))
            for opponent in opinions['against']:
                if opponent in personId and opponent in noidea:
                    continue
                edge_against_index.append((personId[person], personId[opponent]))
        # Constructe edge information (vote)
        for voter, votee in vote_data.items():
            if votee in noidea:
                continue
            edge_vote_index.append((personId[voter], personId[votee]))

        if i == 0:  # the first round
            y = []
            for player in personId.keys():
                if LabelInf[player] == 0:  # folk
                    y.append([0])
                else:
                    y.append([1])  # spy
            data = HeteroData()
            data['players'].x = torch.tensor([xlabel[i] for i in range(len(personId))], dtype=torch.float)  # only one type of node
            data['players'].y = torch.tensor(y)
            # Construct 'for' edges; if the tendency information does not exist, set the shape size to [2,0].
            if len(edge_for_index) == 0:
                data['players', 'for', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'for', 'players'].edge_index = (torch.tensor(edge_for_index, dtype=torch.long).t().contiguous())
            # Construct 'against' edges; if the tendency information does not exist, set the shape size to [2,0].
            if len(edge_against_index) == 0:
                data['players', 'against', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'against', 'players'].edge_index = (torch.tensor(edge_against_index, dtype=torch.long).t().contiguous())
            # Construct 'vote' edges; if the tendency information does not exist, set the shape size to [2,0].
            if len(edge_vote_index) == 0:
                data['players', 'vote', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'vote', 'players'].edge_index = (torch.tensor(edge_vote_index, dtype=torch.long).t().contiguous())
            hetero_data_once.append(data.to(device))
            i = i + 1

        else:  # Two rounds
            data_ref = hetero_data_once[len(hetero_data_once) - 1]
            # copy the graph of first round
            data = HeteroData()
            data['players'].x = data_ref['players'].x
            data['players'].y = data_ref['players'].y
            data['players', 'for', 'players'].edge_index = data_ref['players', 'for', 'players'].edge_index
            data['players', 'against', 'players'].edge_index = data_ref['players', 'against', 'players'].edge_index
            data['players', 'vote', 'players'].edge_index = data_ref['players', 'vote', 'players'].edge_index
            # construct the edges in second round
            if len(edge_for_index) == 0:
                data['players', 'for1', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'for1', 'players'].edge_index = (torch.tensor(edge_for_index, dtype=torch.long).t().contiguous())
            if len(edge_against_index) == 0:
                data['players', 'against1', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'against1', 'players'].edge_index = (torch.tensor(edge_against_index, dtype=torch.long).t().contiguous())
            if len(edge_vote_index) == 0:
                data['players', 'vote1', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'vote1', 'players'].edge_index = (torch.tensor(edge_vote_index, dtype=torch.long).t().contiguous())
            hetero_data_twice.append(data.to(device))

    return hetero_data_once, hetero_data_twice, personId


def CreateDataset(path):
    '''
    Establish heterogeneous graph datasets, represented in the form of a list

    Args:
    path: the path of dataset

    Return:
    The first element is a list, comprising of the HeteroData of the first round of each game
    The second element is a list, comprising of the HeteroData of the game with two rounds
    '''
    dataset_OnceInf = []
    dataset_TwiceInf = []
    validdata = []
    try:
        with open(f'{path}1Avalid.pkl', 'rb') as file:
            validdata = pickle.load(file)
    except FileNotFoundError:
        print('The dataset has not been preprocessed. Please perform preprocessing operations first.')
        exit(0)
    for i in validdata:
        with open(f'{path}tendency_{i}.json', 'r', encoding='utf-8') as file:
            json_tendency = json.load(file)
        with open(f'{path}vote_{i}.json', 'r', encoding='utf-8') as file:
            json_vote = json.load(file)
        with open(f'{path}players_{i}.json', 'r', encoding='utf-8') as file:
            json_player = json.load(file)
        hetero_data_once, hetero_data_twice, _ = json_to_pyg_graph(json_tendency, json_vote, json_player)
        dataset_OnceInf.extend(hetero_data_once)
        dataset_TwiceInf.extend(hetero_data_twice)
    print(f'The length of the dataset of the first rounds of all games is {len(dataset_OnceInf)}, the length of the dataset of games with two rounds is {len(dataset_TwiceInf)}')
    return dataset_OnceInf, dataset_TwiceInf

if __name__ == '__main__':
    CreateDataset('../dataset/meow_8k_095_080/')
