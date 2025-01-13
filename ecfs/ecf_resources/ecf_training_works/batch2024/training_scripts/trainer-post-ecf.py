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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

args = parser.parse_args()

batchsize = args.batch_size
num_nodes = args.starting_nodes
layers = args.num_layers
input_vars = args.num_input_vars

print(f' batch size {batchsize}')
print(f' num nodes {num_nodes}')
print(f' layers {layers}')
print(f' input vars {input_vars}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the cluster and process IDs to create unique filenames and directories
output_dir = f"./outputs_pecf/{num_nodes}nodes_{layers}layers_{input_vars}ecfs_{batchsize}batch_100-150msd"
os.makedirs(output_dir, exist_ok=True)

print('Starting Readins')

hgg = pd.read_parquet('/scratch365/cmoore24/training/data/skimmed_jets/flat400/combed.parquet')
qcd = pd.read_parquet('/scratch365/cmoore24/training/data/skimmed_jets/qcd/q476_goodjets.parquet')



hgg['isSignal'] = 1
qcd['isSignal'] = 0

qcd = pd.concat([hgg, qcd])

del(hgg)


qcd = qcd.select_dtypes(exclude=['object'])
qcd = qcd.drop(columns=qcd.filter(like='Id').columns)
qcd = qcd.drop(columns=['particleNet_mass'])

qcd = qcd[qcd['msoftdrop'] >= 100]
qcd = qcd[qcd['msoftdrop'] <= 150]

feature_names = qcd.columns
feature_names = feature_names[:-1]

X = qcd.drop(columns=['isSignal']).values
y = qcd['isSignal'].values

print('Starting RF Classification')
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = input_vars
X_reduced_rf = X[:, indices[:top_n]]

rf_selected_features = feature_names[indices[:top_n]].tolist()

X_train, X_test, y_train, y_test = train_test_split(X_reduced_rf, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train).type(torch.float).to(device)
X_test = torch.from_numpy(X_test).type(torch.float).to(device)
y_train = torch.from_numpy(y_train).type(torch.float).to(device)
y_test = torch.from_numpy(y_test).type(torch.float).to(device)

# Neural Network Class with Dropout and L2 Regularization
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, width, num_layers, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        current_width = width
        for _ in range(1, num_layers):
            next_width = max(current_width // 2, 1)
            layers.append(nn.Linear(current_width, next_width))
            layers.append(nn.BatchNorm1d(next_width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_width = next_width

        # Output layer
        layers.append(nn.Linear(current_width, 1))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork(input_size=input_vars, width=num_nodes, num_layers=layers).to(device)
model = nn.DataParallel(model, device_ids=[0])
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Scheduler and early stopping
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
early_stop_patience = 20
best_val_loss = float('inf')
early_stop_counter = 0

batch_size = batchsize

# Training loop with early stopping
print('Starting training...')
losses, val_losses = [], []
for epoch in tqdm(range(800)):
    model.train()
    batch_loss = []
    for b in range(0, X_train.size(0), batch_size):
        X_batch = X_train[b: b + batch_size]
        y_batch = y_train[b: b + batch_size].view(-1, 1)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_loss.append(loss.item())
    
    # Training and validation loss
    losses.append(np.mean(batch_loss))
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test)
        val_loss = loss_fn(val_pred, y_test.view(-1, 1)).item()
    val_losses.append(val_loss)

    # Scheduler step and early stopping check
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0  # reset counter
        torch.save(model.state_dict(), "best_model.pth")  # save best model
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print("Training completed.")


with torch.inference_mode():
    # plot loss vs epoch
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(2, 2, 1)
    ax.plot(losses[:], label="loss")
    ax.plot(val_losses[:], label="test_loss")
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

with open(f'{output_dir}/selected_vars.txt', 'w') as f:
    for feature in rf_selected_features:
        f.write(f"{feature}\n")
