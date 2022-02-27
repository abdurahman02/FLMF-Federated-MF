import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
import os

# !git clone "https://github.com/abdurahman02/ml-latest-small.git"
# os.chdir("ml-latest-small")
os.listdir()

data = pd.read_csv("ratings.csv")
data.head()

# split train and validation before encoding
np.random.seed(3)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].copy()
val = data[~msk].copy()

# here is a handy function modified from fast.ai
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

# encoding the train and validation data
df_train = encode_data(train)
df_val = encode_data(val, train)

num_users = len(df_train.userId.unique())
num_items = len(df_train.movieId.unique())
print(num_users, num_items)

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

if torch.cuda.is_available():
  model = MF(num_users, num_items, emb_size=100).cuda()
else:
  model = MF(num_users, num_items, emb_size=100)

def test_loss(model, unsqueeze=False):
    model.eval()
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
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())

def train_epocs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        if torch.cuda.is_available():
            users = torch.LongTensor(df_train.userId.values).cuda()
            items = torch.LongTensor(df_train.movieId.values).cuda()
            ratings = torch.FloatTensor(df_train.rating.values).cuda()
        else:
            users = torch.LongTensor(df_train.userId.values) # .cuda()
            items = torch.LongTensor(df_train.movieId.values) #.cuda()
            ratings = torch.FloatTensor(df_train.rating.values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item()) 
    test_loss(model, unsqueeze)

# Here is what unsqueeze does
ratings = torch.FloatTensor(df_train.rating.values)
print(ratings.shape)
ratings = ratings.unsqueeze(1) # .cuda()
print(ratings.shape)

train_epocs(model, epochs=10, lr=0.1)

train_epocs(model, epochs=15, lr=0.01)

train_epocs(model, epochs=15, lr=0.01)

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

model = MF_bias(num_users, num_items, emb_size=100) #.cuda()

train_epocs(model, epochs=10, lr=0.05, wd=1e-5)

train_epocs(model, epochs=10, lr=0.01, wd=1e-5)

train_epocs(model, epochs=10, lr=0.001, wd=1e-5)