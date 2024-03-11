# -*- coding:utf-8 -*-
import pandas as pd

# def write_to_excel(dataset,random_state,lr,weight_deacy,hidden_channels,hidden_channels1,heads,dropout,best_epoch,train_loss,train_acc,test_loss,test_acc,val_loss,val_acc,patience):    
#     colums=["dataset","patience","random state","lr","weight_deacy","hidden_channels","hidden_channels1","heads","dropout","best_epoch","train_loss","train_acc","val_loss","val_acc","test_loss","test_acc"]
#     # for i in colums:
#     #     print(i,end=',')
#     # exit(0)
#     df = pd.read_excel("./results/record-OneDataset.xlsx")
#     res=[dataset,patience,random_state,lr,weight_deacy,hidden_channels,hidden_channels1,heads,dropout,best_epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc]
#     dataframe = pd.DataFrame([res], columns=colums)
#     df_new = df.append(dataframe, ignore_index=True)
#     df_new.to_excel("./results/record-OneDataset.xlsx", index=False, engine="openpyxl")

def write_to_excel(mode,dataset,patience,random_state,lr,weight_decay,hidden_channels,hidden_channels1,heads,dropout,batch_size,best_epoch1,best_epoch_2,best_epoch3,best_epoch4,train_loss_avg,train_acc_avg,val_loss_avg,val_acc_avg,test_loss_avg,test_acc_avg):
    colums=["model","dataset","patience","random_state","lr","weight_decay","hidden_channels","hidden_channels1","heads","dropout","batch_size","best_epoch1","best_epoch_2","best_epoch3","best_epoch4","train_loss_avg","train_acc_avg","val_loss_avg","val_acc_avg","test_loss_avg","test_acc_avg"]
    res=[mode,dataset,patience,random_state,lr,weight_decay,hidden_channels,hidden_channels1,heads,dropout,batch_size,best_epoch1,best_epoch_2,best_epoch3,best_epoch4,train_loss_avg,train_acc_avg,val_loss_avg,val_acc_avg,test_loss_avg,test_acc_avg]
    df = pd.read_excel("./results/record-4Fold-OneDataset.xlsx")
    dataframe = pd.DataFrame([res], columns=colums)
    df_new = df.append(dataframe, ignore_index=True)
    df_new.to_excel("./results/record-4Fold-OneDataset.xlsx", index=False, engine="openpyxl")    

if __name__=='__main__':
    for i in range(10):
        write_to_excel('6:4:1',1,2,3,4,5,6,7,8,9,10,11,12,13,14)
    # config_OneDataset=[]
    # with open('config_OneDataset.json','r',encoding='utf-8') as file:
    #     config_OneDataset=json.load(file)
    # print(config_OneDataset)
    # print(config_OneDataset['lr'])