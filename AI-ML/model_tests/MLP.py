import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import pyarrow.parquet as pq

# -----------------------------
# 1. Load data from Parquet
# -----------------------------
def load_parquet_dataset(file_path, target_column):
    df = pd.read_parquet(file_path)
    df = df.fillna(0)
    print(df.head(10))
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Example usage
X, y = load_parquet_dataset("../Data/V24-2025__0001.parquet", target_column="TempSaug [°C]")

# -----------------------------
# 2. Prepare DataLoader
# -----------------------------
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# 3. Define a simple MLP model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X.shape[1])

# -----------------------------
# 4. Training setup
# -----------------------------
criterion = nn.MSELoss()  # For regression; use CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5. Training loop
# -----------------------------
for epoch in range(10):
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

print("✅ Training complete.")

