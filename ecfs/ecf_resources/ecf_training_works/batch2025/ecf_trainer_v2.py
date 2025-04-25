import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
import awkward as ak
import dask_awkward as dak
import gc
import math
import warnings

warnings.filterwarnings('ignore', '__array__ implementation')

# Argument parser to get learning rate and batch size
parser = argparse.ArgumentParser(description="Training script for model.")
parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
parser.add_argument('--starting_nodes', type=int, required=True, help='Number of nodes in the first hidden layer')
parser.add_argument('--num_layers', type=int, required=True, help='Number of hidden layers')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
parser.add_argument('--resume', type=str, default=None, help='Path to saved state dict for resuming training')

args = parser.parse_args()
batchsize = args.batch_size
num_nodes = args.starting_nodes
layers = args.num_layers
output_dir = args.output_dir
resume_path = args.resume

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preparation (simplified for brevity)
hgg = ak.from_parquet('/scratch365/cmoore24/training/hgg/batch2025/inputs/flat_ratios.parquet', columns='train_ratios').train_ratios
qcd = ak.from_parquet('/scratch365/cmoore24/training/hgg/batch2025/inputs/qcd_reduced_ratios.parquet', columns='train_ratios').train_ratios

feature_names = qcd.fields
hgg_ones = ak.ones_like(hgg[hgg.fields[0]])
qcd_zeros = ak.zeros_like(qcd[qcd.fields[0]])
combined = ak.concatenate([hgg, qcd])

X = np.column_stack([ak.to_numpy(combined[feature]) for feature in feature_names])
y = np.array(ak.concatenate([hgg_ones, qcd_zeros]))

del(hgg)
del(qcd)
del(hgg_ones)
del(qcd_zeros)
del(combined)
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler().fit(X_train)
X_train = torch.from_numpy(scaler.transform(X_train)).float().to(device)
X_test = torch.from_numpy(scaler.transform(X_test)).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

# Neural Network definition
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, width, num_layers, dropout_rate=0.3):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_size, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(1, num_layers):
            next_width = max(width // 2, 1)
            layers.extend([nn.Linear(width, next_width), nn.BatchNorm1d(next_width), nn.ReLU(), nn.Dropout(dropout_rate)])
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

model = NeuralNetwork(len(feature_names), num_nodes, layers).to(device)
model = nn.DataParallel(model, device_ids=[0,1,2,3], output_device=0)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-7)

# Resume from saved state
best_val_loss = float('inf')
if resume_path:
    checkpoint = torch.load(f'{resume_path}/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['best_val_loss']

    with open(f'{output_dir}/losses.pkl', "rb") as f:
        losses, val_losses = pickle.load(f)
else:
    losses, val_losses = [], []
    best_val_loss = float('inf')

# Training loop

for epoch in tqdm(range(2000)):
    model.train()
    batch_loss = []
    for b in range(0, X_train.size(0), batchsize):
        X_batch = X_train[b:b + batchsize]
        y_batch = y_train[b:b + batchsize].view(-1, 1)
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    losses.append(np.mean(batch_loss))
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_test), y_test.view(-1, 1)).item()
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    
    # Save loss plot
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f'{output_dir}/loss_curve_epoch.png')
    plt.close()

    # Save the best model and state
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss}, os.path.join(output_dir, "best_model.pth"))
        traced_model = torch.jit.trace(model.module.cpu(), X_test.cpu())
        traced_model.save(os.path.join(output_dir, "traced_model.pt"))
        with open(f'{output_dir}/losses.pkl', "wb") as f:
            pickle.dump((losses, val_losses), f)
        model.module.to(device)



    if epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")