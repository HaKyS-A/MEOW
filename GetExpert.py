import json
import os

import torch
from torch_geometric.data import HeteroData

from utils.util import set_random_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def call_expert(cur_round: int, tendency: list, vote: list, random_state: int):
    '''
    get the expert observation for a game
    The best model is stored in mode/bestmodel/.
    For the first round, the corresponding model is named 'OneRound-random-i-bestmodel-j' (where i corresponds to the random_state and j corresponds to the fold).
    For the second round, the corresponding model is named 'TwoRound--random-i-bestmodel-j'

    Args:
    cur_round: current round. In our experiment, cur_round in [1,2]
    tendency：[ {'players1':{'for':['players'],'against':['players'],'no comment':[''players]},'players2':{...},...},...]
    vote:[{'players1':'players',...}]
    random_state: determine the expert model, 1-10

    Return:
    return the name of the spy in expert observation
    '''
    bestmodel_path = './model/bestmodel/'
    assert cur_round in [1, 2], 'round should in [1,2]'
    assert random_state in range(1, 11), 'random_state should between 1 and 10'
    xlabel = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    personId = {}
    # etrieve players and their id
    n = 0
    for i in vote[0]:
        personId[i] = n
        n += 1
    noidea = ['', '无', '无法选择', '不投票']
    # construct heterogeneous graph
    data = HeteroData()  # 异构图Data
    data['players'].x = torch.tensor([xlabel[i] for i in range(len(personId))], dtype=torch.float)
    votecnt = {}  # vote[a]=the number of votes for a in the first round
    for i in range(cur_round):
        # Constructe edge information (for or against)
        edge_for_index = []
        edge_against_index = []
        edge_vote_index = []
        for person, opinions in tendency[i].items():
            for supporter in opinions['for']:
                if supporter in personId and supporter in noidea:
                    continue
                edge_for_index.append((personId[person], personId[supporter]))
            for opponent in opinions['against']:
                if opponent in personId and opponent in noidea:
                    continue
                edge_against_index.append((personId[person], personId[opponent]))
        # construct edge information (vote)
        for voter, votee in vote[i].items():
            if votee in noidea:
                continue
            if i == 0:
                if votee not in votecnt:
                    votecnt[votee] = 1
                else:
                    votecnt[votee] += 1
            edge_vote_index.append((personId[voter], personId[votee]))
        if i == 0:
            # if the tendency information does not exist, set the shape size to [2,0].
            if len(edge_for_index) == 0:
                data['players', 'for', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'for', 'players'].edge_index = (torch.tensor(edge_for_index, dtype=torch.long).t().contiguous())
            if len(edge_against_index) == 0:
                data['players', 'against', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'against', 'players'].edge_index = (torch.tensor(edge_against_index, dtype=torch.long).t().contiguous())
            if len(edge_vote_index) == 0:
                data['players', 'vote', 'players'].edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                data['players', 'vote', 'players'].edge_index = (torch.tensor(edge_vote_index, dtype=torch.long).t().contiguous())
        else:
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
    data = data.to(device)
    path = f'{bestmodel_path}OneRound-random-{random_state}-bestmodel-'
    output = torch.tensor([[], []])
    name = list(personId.keys())  # name[i]=the name of player_i
    noid = -1  # The eliminated player must be folk
    if cur_round == 2:
        path = f'{bestmodel_path}TwoRound-random-{random_state}-bestmodel-'
        # In the case of a tie, the person who received the first vote earlier will be eliminated, the order of keywords in a set is consistent with the order of receiving votes
        rec = 0
        for per, val in votecnt.items():
            if val > rec:
                noid = per
                rec = val
    for i in range(4):
        model = torch.load(f'{path}{i}.pth')
        if i == 0:
            output = model(data.x_dict, data.edge_index_dict)
        else:
            output += model(data.x_dict, data.edge_index_dict)
    result = torch.argmax(output, dim=1)
    if result == noid:
        # Why do this instead of setting it to 0? Because in some data, the resulting vector may be all negative. Setting it to 0 would result in the eliminated person becoming spy
        output[0][noid] = 0
        result = torch.argmax(output, dim=1)
    # print(f'The Expert think {name[result]} is the undercover!')
    return name[result]
