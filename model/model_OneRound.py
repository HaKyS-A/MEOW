import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv


class GAT_OneRoud(torch.nn.Module):
    '''
    The GATv2 model is used to train the dataset of the first round of each game. 3 edge types ['players','for'/'against'/'vote','players']
    '''

    def __init__(self,
        hidden_channels,
        hidden_channels1,
        heads,
        out_channels,
        aggr_way,
        dropvalue,
    ):
        '''
        hidden_channels: The output channel of the first GATv2Conv layer
        hidden_channels: The output channel of the second GATv2Conv layer
        heads: The number of attention heads
        out_channels: The final output channel, in our experiments, it's 4
        aggr_way: The aggregation scheme to use for grouping node embeddings generated by different relations
        dropvalue:  The probability of setting an element to zero during the dropout operation
        '''
        super().__init__()

        self.conv1 = HeteroConv(
            {
                ('players', 'for', 'players'): GATv2Conv(
                    (-1, -1), hidden_channels, heads=heads, add_self_loops=False
                ),
                ('players', 'against', 'players'): GATv2Conv(
                    (-1, -1), hidden_channels, heads=heads, add_self_loops=False
                ),
                ('players', 'vote', 'players'): GATv2Conv(
                    (-1, -1), hidden_channels, heads=heads, add_self_loops=False
                ),
            },
            aggr=aggr_way,
        )  # hidden_channels*heads
        self.conv2 = HeteroConv(
            {
                ('players', 'for', 'players'): GATv2Conv(
                    (-1, -1), hidden_channels1, heads=heads, add_self_loops=False
                ),
                ('players', 'against', 'players'): GATv2Conv(
                    (-1, -1), hidden_channels1, heads=heads, add_self_loops=False
                ),
                ('players', 'vote', 'players'): GATv2Conv(
                    (-1, -1), hidden_channels1, heads=heads, add_self_loops=False
                ),
            },
            aggr=aggr_way,
        )  # hidden_channels1*heads
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropvalue)
        self.fc = nn.Linear(hidden_channels1 * heads * 4, out_channels)

    def forward(self, x, edge_index):
        '''
        x: A dictionary holding node feature information for each individual node type. In our experiment, there is only one node type ('players').
        edge_index_dict: A dictionary holding graph connectivity information for each individual edge type
        '''
        # x['players']:(batchsize*4[four players->four nodes],4)
        x = self.conv1(x, edge_index)
        # x=defaultdict(<class 'list'>, {'players':tensor(batchsize*4,hidden_channels*heads)}
        x = self.conv2(x, edge_index)
        # x=defaultdict(<class 'list'>, {'players':tensor(batchsize*4,hidden_channels1*heads)}
        x = x['players']
        # x=tensor(batchsize*4,hidden_channels1*heads)
        x = x.reshape(-1, x.shape[1] * 4)
        # (batchsize*4,hidden_channels1*heads)->(batchsize,4*hidden_channels1*heads),Each row concatenates the respective information of four players
        x = self.relu(x)
        # (batchsize,4*hidden_channels1*heads)
        x = self.dropout(x)
        x = self.fc(x)
        # （batchsize,outchannels=4)
        return x
