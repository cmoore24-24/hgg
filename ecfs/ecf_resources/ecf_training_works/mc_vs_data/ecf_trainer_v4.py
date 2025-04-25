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
import gc
import warnings
from torch.utils.data import Dataset, DataLoader

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

gpus=[0]#,1,2,3]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and len(gpus) > 0:
    device = torch.device(f'cuda:{gpus[0]}')
    torch.cuda.set_device(gpus[0])
else:
    device = torch.device("cpu")
    gpus = []

def log_gpu_utilization():
    for i in range(torch.cuda.device_count()):
        utilization = torch.cuda.utilization(i)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
        print(f"GPU {i}: Utilization={utilization}%, Allocated={memory_allocated:.2f}MB, Reserved={memory_reserved:.2f}MB")

mc = ak.from_parquet('/scratch365/cmoore24/training/hgg/ecf_vs_data/inputs/train/mc.parquet')
data = ak.from_parquet('/scratch365/cmoore24/training/hgg/ecf_vs_data/inputs/train/data.parquet')

feature_names = mc.fields
mc_ones = ak.ones_like(mc[mc.fields[0]])
data_zeros = ak.zeros_like(data[data.fields[0]])
combined = ak.concatenate([mc, data])

X = np.column_stack([ak.to_numpy(combined[feature]) for feature in feature_names])
y = np.array(ak.concatenate([mc_ones, data_zeros]))

del(mc)
del(data)
del(mc_ones)
del(data_zeros)
del(combined)
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

# Define Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets and data loaders
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=1,
                          pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=1)

# Neural Network definition
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, width, num_layers, dropout_rate=0.05):
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
model = nn.DataParallel(model, device_ids=gpus, output_device=0)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, min_lr=1e-7)

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

# Training loop
for epoch in tqdm(range(2000)):
    model.train()
    batch_loss = []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True).view(-1, 1)


        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        batch_loss.append(loss.item())

    losses.append(np.mean(batch_loss))

    # Validation step
    model.eval()
    val_loss = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).view(-1, 1)
            loss = loss_fn(model(X_batch), y_batch)
            val_loss.append(loss.item())
    val_losses.append(np.mean(val_loss))

    scheduler.step(np.mean(val_loss))
    print(log_gpu_utilization(), flush=True)

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
    if np.mean(val_loss) < best_val_loss:
        best_val_loss = np.mean(val_loss)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss}, os.path.join(output_dir, "best_model.pth"))
        traced_model = torch.jit.trace(model.module.cpu(), torch.randn(1, len(feature_names)))
        traced_model.save(os.path.join(output_dir, "traced_model.pt"))
        with open(f'{output_dir}/losses.pkl', "wb") as f:
            pickle.dump((losses, val_losses), f)
        model.module.to(device)

    if epoch % 5 == 0:
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}: Learning Rate = {param_group['lr']}")
