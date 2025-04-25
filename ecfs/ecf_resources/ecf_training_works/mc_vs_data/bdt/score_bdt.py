import time
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import pickle
import argparse
import awkward as ak
import gc
import warnings

warnings.filterwarnings('ignore', '__array__ implementation')

# Argument parser for training parameters
parser = argparse.ArgumentParser(description="BDT Training Script.")
parser.add_argument('--num_trees', type=int, required=True, help='Number of trees for XGBoost')
parser.add_argument('--max_depth', type=int, required=True, help='Max depth for each tree')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping rounds')
args = parser.parse_args()

num_trees = args.num_trees
max_depth = args.max_depth
output_dir = args.output_dir
early_stopping_rounds = args.early_stopping

print('Reading Files...')

# with open('/scratch365/cmoore24/training/hgg/batch2025/ml_results_check/sub_ratios/1291.pkl', 'rb') as f:
#     ecfs = pickle.load(f)

mc = ak.from_parquet('/scratch365/cmoore24/training/hgg/ecf_vs_data/inputs/train/mc.parquet')
data = ak.from_parquet('/scratch365/cmoore24/training/hgg/ecf_vs_data/inputs/train/data.parquet')

ecfs = mc.fields
print(f'There are {len(ecfs)} features')

mc_ones = ak.ones_like(mc[mc.fields[0]])
data_zeros = ak.zeros_like(data[data.fields[0]])
combined = ak.concatenate([mc, data])

X = np.column_stack([ak.to_numpy(combined[feature]) for feature in ecfs])
print(f'There are now {len(X[0])} features')
y = np.array(ak.concatenate([mc_ones, data_zeros]))

del mc, data, mc_ones, data_zeros, combined
gc.collect()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

# Convert data into DMatrix format (optimized for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=ecfs)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=ecfs)

# Define XGBoost Parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": max_depth,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "tree_method": "hist",  # Use "hist" instead of "gpu_hist"
    "device": "cuda",       # Explicitly set to GPU
    "verbosity": 2
}

# Train BDT Model with Early Stopping
print("Training XGBoost BDT...")
evals_result = {}  # To store evaluation metrics over training
start_time = time.time()
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_trees,
    evals=[(dtest, "eval")],  # Monitor validation loss
    evals_result=evals_result,
    # early_stopping_rounds=early_stopping_rounds,
    verbose_eval=10  # Print loss every 10 trees
)
end_time = time.time()

# Log training time
print(f"\nTraining completed in {end_time - start_time:.4f} seconds")

# Evaluate Model
y_pred_proba = bst.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Save Model
bst.save_model(os.path.join(output_dir, "bdt_model.json"))
print("\nBDT Model saved successfully!")

# Save Log Loss Plot
log_loss_values = evals_result["eval"]["logloss"]

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(log_loss_values) + 1), log_loss_values, marker='o', linestyle='-')
plt.xlabel("Number of Trees")
plt.ylabel("Log Loss")
plt.title("XGBoost Log Loss Over Trees")
plt.grid()
plt.savefig(os.path.join(output_dir, "bdt_loss_curve.png"))
plt.show()
plt.close()
