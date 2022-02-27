import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os
import copy
import time
import string
import random


def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df


def filtering_data(df,from_user, to_user, from_item, to_item):
  if(from_user <= to_user and from_item <= to_item
     and to_user < max(df["userId"]) and to_item < max(df["movieId"])
     ):
    return df[(df.userId >= from_user) & 
              (df.userId <= to_user) &
              (df.movieId >= from_item) &
              (df.movieId <= to_item)
              ]
  print("Error Range")

def getBatchForUser(data, u, batchSize):
  if u >= len(data["userId"].unique()):
    print("INvalid UserId requested")
    return
  if batchSize > len(data[data.userId == u]):
    batchSize = len(data[data.userId == u])
  return data[data.userId == u].sample(n=batchSize)

def add_model_parameters(model1, model2):
    # Adds the parameters of model1 to model2

    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
        #   if name1 != 'userX_embedding.weight':
            dict_params2[name1].data.copy_(param1.data + dict_params2[name1].data)

    model2.load_state_dict(dict_params2)

def sub_model_parameters(model1, model2):
    # Subtracts the parameters of model2 with model1

    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
        #   if name1 != 'userX_embedding.weight':
            dict_params2[name1].data.copy_(dict_params2[name1].data - param1.data)

    model2.load_state_dict(dict_params2)

def divide_model_parameters(model, f):
    # Divides model parameters except for the user embeddings with f
    params1 = model.named_parameters()
    params2 = model.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 != 'user_emb.weight':
            dict_params2[name1].data.copy_(param1.data / f)
    model.load_state_dict(dict_params2)

def zero_model_parameters(model):
    # sets all parameters to zero

    params1 = model.named_parameters()
    params2 = model.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data - dict_params2[name1].data)

    model.load_state_dict(dict_params2)

def AssignUserEmbedding(model, user_emb):
  params1 = model.named_parameters()
  dict_params1 = dict(params1)
  dict_params1['userX_embedding.weight'].data = copy.deepcopy(user_emb)
  model.load_state_dict(dict_params1)

def printModelParam(model):
  print("hello world")
  params1 = model.named_parameters()
  print(params1)
  dict_params1 = dict(params1)
  print(dict_params1)
  print(dict_params1['userX_embedding.weight'].data)


class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        
    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u*v).sum(1)

class MF02(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF02, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        # self.user_emb.weight.data.uniform_(0, 0.05)
        # self.item_emb.weight.data.uniform_(0, 0.05)
        
    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u*v).sum(1)



def test_loss(model01, df_val, unsqueeze=False):
    model01.eval()
    if torch.cuda.is_available():
      users = torch.LongTensor(df_val.userId.values).cuda()
      items = torch.LongTensor(df_val.movieId.values).cuda()
      ratings = torch.FloatTensor(df_val.rating.values).cuda()
    else:
      users = torch.LongTensor(df_val.userId.values)
      items = torch.LongTensor(df_val.movieId.values)
      ratings = torch.FloatTensor(df_val.rating.values)
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model01(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())
    return loss.item()



def train_epocs(model_server,df_train, df_val, userBatchSize, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    train_loss = []
    model_diff = copy.deepcopy(model_server)
    if torch.cuda.is_available():
      model_diff.cuda()
    zero_model_parameters(model_diff)

    t1 = time.time()
    
    # following line of code is for selecting random users from training data
    User_batch = random.sample(list(df_train.userId.unique()), userBatchSize)


    for user_id in User_batch:

        model02 = copy.deepcopy(model_server)
        model02.train()
        if torch.cuda.is_available():
            model02.cuda()
        optimizer = torch.optim.Adam(model02.parameters(), lr=lr, weight_decay=wd)
    
        sel_rows_of_user_i = df_train[(df_train.userId == user_id)]
        if torch.cuda.is_available():
            users = torch.LongTensor(sel_rows_of_user_i.userId.values).cuda()
            items = torch.LongTensor(sel_rows_of_user_i.movieId.values).cuda()
            ratings = torch.FloatTensor(sel_rows_of_user_i.rating.values).cuda()
        else:
            users = torch.LongTensor(sel_rows_of_user_i.userId.values) # .cuda()
            items = torch.LongTensor(sel_rows_of_user_i.movieId.values) #.cuda()
            ratings = torch.FloatTensor(sel_rows_of_user_i.rating.values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)

        for epoch in range(1):
            y_hat = model02(users, items)
            loss = F.mse_loss(y_hat, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.item())
            train_loss.append(loss.item())

        sub_model_parameters(model_server, model02)
        add_model_parameters(model02, model_diff)

    # Take the average of the MLP and item vectors
    divide_model_parameters(model_diff, (len(df_train.userId.unique())))

    # Update the global model by adding the total change
    add_model_parameters(model_diff, model_server)
    
    t2 = time.time()
    print("Time of round:", round(t2 - t1), "seconds")

    
    return test_loss(model_server, df_val, unsqueeze), np.mean(train_loss)



def main():

    test_loss_vec = []
    train_loss_vec = []
    userBatchSize = 200
    rounds=10

    os.chdir("ml-latest-small")
    os.listdir()

    data = pd.read_csv("ratings.csv")
    data.head()

    msk = np.random.rand(len(data)) < 0.8
    train = data[msk].copy()
    val = data[~msk].copy()

    # encoding the train and validation data
    df_train = encode_data(train)
    df_val = encode_data(val, train)

    num_users = len(df_train.userId.unique())
    num_items = len(df_train.movieId.unique())
    print(num_users, num_items)


    
    if torch.cuda.is_available():
      model_s = MF(num_users, num_items, emb_size=100).cuda()
    else:
        model_s = MF(num_users, num_items, emb_size=100)

    for t in range(10):
        testLoss,trainLoss = train_epocs(model_s,df_train, df_val, userBatchSize, epochs=10, lr=0.1)
        test_loss_vec.append(testLoss)
        train_loss_vec.append(trainLoss)

    for t in range(20):
        testLoss,trainLoss = train_epocs(model_s,df_train, df_val, userBatchSize, epochs=15, lr=0.01)
        test_loss_vec.append(testLoss)
        train_loss_vec.append(trainLoss)

    os.chdir("..")
    os.chdir("FLMF_res")
    filename="results09.csv"
    total_round = [x for x in range(len(test_loss_vec))]
    headers = ['round', 'Mean_Train_Loss', 'Test_Loss']
    with open(filename, 'w') as file:
        for header in headers:
            file.write(str(header)+', ')
        file.write('\n')
        for i in range(len(test_loss_vec)):
            file.write(str(total_round[i])+', ')
            file.write(str(train_loss_vec[i])+', ')
            file.write(str(test_loss_vec[i])+', ')
            file.write('\n')

    file.close()


    data_results= pd.read_csv("results09.csv")
    print("data from results: \n",data_results)
    
 

main()



class MF_bias(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF_bias, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)
        
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        return (U*V).sum(1) +  b_u  + b_v

