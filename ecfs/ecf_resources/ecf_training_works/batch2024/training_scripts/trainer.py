print('The script has started')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse

# Argument parser to get learning rate and batch size
parser = argparse.ArgumentParser(description="Training script for model.")
parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
parser.add_argument('--starting_nodes', type=int, required=True, help='Number of nodes in the first hidden layer')
parser.add_argument('--num_layers', type=int, required=True, help='Number of hidden layers')
parser.add_argument('--num_input_vars', type=int, required=True, help='Number of ECFs to use as input')
parser.add_argument('--msoftdrop_formation', type=int, required=True, help='0 = No sculpting, 1 = Hgg sculpted, 2 = QCD sculpted')

args = parser.parse_args()

batchsize = args.batch_size
num_nodes = args.starting_nodes
layers = args.num_layers
input_vars = args.num_input_vars
data_formation = args.msoftdrop_formation

print(f' batch size {batchsize}')
print(f' num nodes {num_nodes}')
print(f' layers {layers}')
print(f' input vars {input_vars}')
print(f' data formation {data_formation}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the cluster and process IDs to create unique filenames and directories
output_dir = f"./outputs/{num_nodes}nodes_{layers}layers_{input_vars}ecfs_{batchsize}batch_{data_formation}msoftdrop/"
os.makedirs(output_dir, exist_ok=True)

print('Starting Readins')

if data_formation == 0:    
    hgg = pd.read_parquet('/scratch365/cmoore24/training/hgg/batch/inputs/hgg_ratios_array_small.parquet')
    qcd = pd.read_parquet('/scratch365/cmoore24/training/hgg/batch/inputs/qcd_ratios_array_small.parquet')
    with open('/scratch365/cmoore24/training/hgg/batch/inputs/selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
elif data_formation == 1:
    hgg = pd.read_parquet('/scratch365/cmoore24/training/hgg/batch/inputs/hgg_sculpted_ratios_array_small.parquet')
    qcd = pd.read_parquet('/scratch365/cmoore24/training/hgg/batch/inputs/qcd_ratios_array_small.parquet')
    with open('/scratch365/cmoore24/training/hgg/batch/inputs/selected_hgg_sculpted_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
elif data_formation == 2:
    hgg = pd.read_parquet('/scratch365/cmoore24/training/hgg/batch/inputs/hgg_ratios_array_small.parquet')
    qcd = pd.read_parquet('/scratch365/cmoore24/training/hgg/batch/inputs/qcd_sculpted_ratios_array_small.parquet')
    with open('/scratch365/cmoore24/training/hgg/batch/inputs/selected_qcd_sculpted_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)

selected_features = selected_features[:input_vars]
hgg = hgg[selected_features]
qcd = qcd[selected_features]

hgg['isSignal'] = 1
qcd['isSignal'] = 0

qcd = pd.concat([hgg, qcd])

del(hgg)

X = qcd.drop(columns=['isSignal']).values
y = qcd['isSignal'].values
del(qcd)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train).type(torch.float).to(device)
X_test = torch.from_numpy(X_test).type(torch.float).to(device)
y_train = torch.from_numpy(y_train).type(torch.float).to(device)
y_test = torch.from_numpy(y_test).type(torch.float).to(device)

###################################

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, width, num_layers):
        super(NeuralNetwork, self).__init__()

        layers = []

        # First hidden layer (input -> first hidden layer with width nodes)
        layers.append(nn.Linear(input_size, width))
        layers.append(nn.ReLU())

        # Add subsequent hidden layers, halving the number of nodes in each layer
        current_width = width
        for _ in range(1, num_layers):
            next_width = current_width // 2
            layers.append(nn.Linear(current_width, next_width))
            layers.append(nn.ReLU())
            current_width = next_width

        # Output layer (connects final hidden layer to the single output node)
        layers.append(nn.Linear(current_width, 1))

        # Store the layers in a Sequential container
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# Example usage:
# input_size should match the number of input features
# width is the number of nodes in the first hidden layer
# num_layers controls how many hidden layers the network will have
model = NeuralNetwork(input_size=input_vars, width=num_nodes, num_layers=layers).to(device)

model = nn.DataParallel(model, device_ids=[0])

loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, verbose=True, threshold=0.1)

###################################

portion = batchsize

print('The training is starting')

losses, test_losses = [], []

batch_size = portion
for t in tqdm(range(30)):
    batch_loss, batch_test_loss = [], []
    
    for b in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[b : b + batch_size]
        y_batch = y_train[b : b + batch_size]
        model.train()
        y_pred = model(X_batch)
        y_b = y_batch.view(-1,1)

        loss = loss_fn(y_pred, y_b)
        batch_loss.append(loss.item())#/batch_size)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.inference_mode():

            model.eval()
            test_pred = model(X_test)

            test_label = y_test.view_as(test_pred)
            test_loss = loss_fn(test_pred, test_label)
            batch_test_loss.append(test_loss.item())#/len(X_test))
    
    losses.append(np.mean(batch_loss))
    test_losses.append(np.mean(batch_test_loss))
    scheduler.step(test_loss)

###################################

with torch.inference_mode():
    # plot loss vs epoch
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(2, 2, 1)
    ax.plot(losses[:], label="loss")
    ax.plot(test_losses[:], label="test_loss")
    ax.legend(loc="upper right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    #ax.set_ylim(0.6, 0.64)

    # Plot ROC
    X_test_in = X_test
    Y_predict = model(X_test_in)
    from sklearn.metrics import roc_curve, auc

    tpr, fpr, thresholds = roc_curve(y_test.cpu(), Y_predict.cpu())
    roc_auc = auc(tpr, fpr)
    ax = plt.subplot(2, 2, 3)
#     ax.set_yscale("log")
    ax.plot(tpr, fpr, lw=2, color="cyan", label="auc = %.3f" % (roc_auc))
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle="--", lw=2, color="k", label="random chance")
    ax.set_xlim([0, 1.0])
#     ax.set_ylim([1e-5, 1.0])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Hgg receiver operating curve")
    ax.legend(loc="lower right")
    ax.axvline(x=0.5, color='black')
    plt.show()
plt.savefig(f'{output_dir}/training_plots.png')

with open(f'{output_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

traced = torch.jit.trace(model.module.cpu(), X_test.cpu())
traced.save(f'{output_dir}/traced_model.pt')

with open(f'{output_dir}/selected_ecfs.txt', 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")
