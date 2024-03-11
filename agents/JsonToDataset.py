import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import json
import pickle

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
# device='cpu'

datafrom=0
#agents文件加下运行
path='./logs/meow_8k_095_080/logs/'
if datafrom==1:
    path='./logs/meow_turbo_075_090/logs/'
elif datafrom==2:
    path='./logs/meow_turbo_095_080/logs/'

def json_to_pyg_graph(json_tendency,json_vote,json_player):
    # graph_data_list = []
    hetero_data = []
    i=0
    xlabel=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    LabelInf={} #LabelInf[player]=identity,eq:LabelInf['Alice']=0【好人】/1【坏人】
    personId={}
    # print(personId)
    #读取json_player中的角色信息
    for i in range(len(json_player[0])):
        LabelInf[json_player[i+1][0]]=json_player[i+1][3]
        personId[json_player[i+1][0]]=i
    i=0
    for round_data,vote_data in zip(json_tendency,json_vote):#每一轮次投票信息
        # 构建边的信息(赞同与否)
        edge_for_index = []
        edge_against_index = []
        edge_vote_index=[]
        for person, opinions in round_data.items():
            for supporter in opinions['for']:
                if supporter in personId and (supporter=="" or supporter=="无" or supporter=="无法选择"):
                    continue
                edge_for_index.append((personId[person], personId[supporter]))
                # edge_for_type.append(1)  # 1 表示赞成
            for opponent in opinions['against']:
                if opponent in personId and (opponent=="" or opponent=="无" or opponent=="无法选择"):
                    continue
                # print(opponent)
                edge_against_index.append((personId[person], personId[opponent]))
                # edge_type.append(-1)  # -1 表示反对
        #构建边的信息（投票
        for voter,votee in vote_data.items():
            if votee=="不投票" or votee=="" or votee=="无" or votee=="无法选择":
                continue
            edge_vote_index.append((personId[voter],personId[votee]))
            # edge_type.append(-2)#-2表示投票给他，因为-1表示怀疑（投票表示最怀疑
        
        if i==0:#一轮次信息
            y=[]
            for player in personId.keys():
                if LabelInf[player]==0:#好人是0
                    y.append([0])
                else:
                    y.append([1])#坏人是1
            data=HeteroData()#这个是一定有的
            data['players'].x=torch.tensor([xlabel[i] for i in range(len(personId))],dtype=torch.float)#只有一个节点
            data['players'].y=torch.tensor(y)
            #赞同边
            data['players','for','players'].edge_index=torch.tensor(edge_for_index, dtype=torch.long).t().contiguous()
            # data['players','for','players'].edge_attr=torch.tensor([[1] for _ in range(len(edge_for_index))])
            #反对边
            data['players','against','players'].edge_index=torch.tensor(edge_against_index, dtype=torch.long).t().contiguous()
            # data['players','against','players'].edge_attr=torch.tensor([[-1] for _ in range(len(edge_against_index))])
            #投票边
            data['players','vote','players'].edge_index=torch.tensor(edge_vote_index, dtype=torch.long).t().contiguous()
            # data['players','vote','players'].edge_attr=torch.tensor([[-2] for _ in range(len(edge_vote_index))])
            # 添加到 HeteroData 中
            hetero_data.append(data.to(device))
            i=i+1
            
            #debug
            # print(data)
            # print("y=",data['players'].y)
            # print("x=",data['players'].x,"\n,edge_for_index=",data['players','for','players'].edge_index)
            # print("against_index=",data['players','against','players'].edge_index)
            # print("vote_index=",data['players','vote','players'].edge_index)
            
        else:#i=1
            # data=hetero_data_once[len(hetero_data_once)-1]#在前面一个的基础上,不行，好像是共用一份内存空间
            # data_ref=hetero_data_once[len(hetero_data_once)-1]
            data=hetero_data[0]
            #赞同边
            data['players','for1','players'].edge_index=torch.tensor(edge_for_index, dtype=torch.long).t().contiguous()
            #反对边
            data['players','against1','players'].edge_index=torch.tensor(edge_against_index, dtype=torch.long).t().contiguous()
            #投票边
            data['players','vote1','players'].edge_index=torch.tensor(edge_vote_index, dtype=torch.long).t().contiguous()
            hetero_data[0]=data.to(device)
            i=i+1
            
    if i==1:
        #补充一下空边
        data=hetero_data[0]
        #赞同边
        data['players','for1','players'].edge_index=torch.tensor([[],[]],dtype=torch.int)
        #反对边
        data['players','against1','players'].edge_index=torch.tensor([[],[]],dtype=torch.int)
        #投票边
        data['players','vote1','players'].edge_index=torch.tensor([[],[]],dtype=torch.int)
        hetero_data[0]=data.to(device)
        i=i+1      
    
    #debug
    # data=hetero_data[0]
    # print(data)
    # print("y=",data['players'].y)
    # print("x=",data['players'].x,"\n,edge_for_index=",data['players','for','players'].edge_index)
    # print("for1_index=",data['players','for1','players'].edge_index)
    # print("against_index=",data['players','against','players'].edge_index)
    # print("against1_index=",data['players','against1','players'].edge_index)
    # print("vote_index=",data['players','vote','players'].edge_index)
    # print("vote1_index=",data['players','vote1','players'].edge_index) 
    # input()
    
    
    return hetero_data,personId

def CreateDataset():
    dataset=[]
    validdata=[]
    with open(path+'1Avalid.pkl','rb') as file:
        validdata=pickle.load(file)
    for i in validdata:
        # i=23
        # print(i)
        # 读取信息
        with open(path+'tendency_'+str(i)+'.json', 'r') as file:
            json_tendency = json.load(file)
        # print(json_tendency)
        with open(path+'vote_'+str(i)+'.json','r') as file:
            json_vote=json.load(file)
        # print(json_vote)
        with open(path+'players_'+str(i)+'.json','r',encoding='utf-8') as file:
            json_player=json.load(file)
        hetero_data,_ = json_to_pyg_graph(json_tendency,json_vote,json_player)
        dataset.extend(hetero_data)
        # print(i)
    #下面是测试用的伪造数据代码
    # for i in range(100):
    #     dataset_OnceInf.append(dataset_OnceInf[0])
    #     dataset_TwiceInf.append(dataset_TwiceInf[0])
    print(len(dataset))
    # input()
    return dataset

if __name__=='__main__':
    dataset=CreateDataset()