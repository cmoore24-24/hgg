import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dask
import dask.dataframe as ddf
import numpy as np
from tqdm import tqdm
import pickle
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

with open('combed_qcd.pkl', 'rb') as f:
    background = pickle.load(f)

with open('./data/signal_vars.pkl', 'rb') as f:
    signal = pickle.load(f)
hgg = signal['hgg']
del(signal)
gc.collect()


var_keys = list(hgg.keys())

items = [
#     'FatJets.area',
#     'FatJets.eta',
#     'FatJets.mass',
#     'FatJets.msoftdrop',
#     'FatJets.n2b1',
#     'FatJets.n3b1',
#     'FatJets.phi',
#     'FatJets.pt',
#     'FatJets.u1',
    'FatJets.m2',
    'isSignal'
]

hgg['isSignal'] = np.ones_like(hgg[items[0]])
background['isSignal'] = np.zeros_like(background[items[0]])

sizes = []
for i in background:
    sizes.append(len(background[i]))

losses = [(x - min(sizes)) for x in sizes]
for i in background:
    background[i] = background[i][:min(sizes)]

background = pd.DataFrame.from_dict(background)

small_sig = hgg[items]
small_bkg = background[items]
del(background)
gc.collect()

NDIM = len(items) - 1
df_all = pd.concat([small_sig, small_bkg])
dataset = df_all.values
X = dataset[:, 0:NDIM]
y = dataset[:, NDIM]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train).type(torch.float).to(device)
X_test = torch.from_numpy(X_test).type(torch.float).to(device)
y_train = torch.from_numpy(y_train).type(torch.float).to(device)
y_test = torch.from_numpy(y_test).type(torch.float).to(device)

# Build our model.
gc.collect()
torch.cuda.empty_cache()

class NeuralNetwork(nn.Module):
    def __init__(self, width=2):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

model= nn.DataParallel(model,device_ids = [0, 1, 2, 3])
model.to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


