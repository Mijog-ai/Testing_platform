import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
)

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
def load_parquet_dataset(file_path, target_column):
    df = pd.read_parquet(file_path).fillna(0)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    return X, y

file_path = "../Data/V24-2025__0001.parquet"
target_column = "TempSaug [°C]"  # replace if your classification label differs

X, y = load_parquet_dataset(file_path, target_column)

# Optional scaling (important for KNN and MLP)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2. Define MLP Classifier
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
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

# -----------------------------
# 3. Train MLP
# -----------------------------
input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
mlp = MLP(input_dim, hidden_dim=64, output_dim=num_classes)

print(mlp)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
), batch_size=32, shuffle=True)

for epoch in range(15):
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")

torch.save(mlp.state_dict(), "mlp_model.pth")
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 4. Train Random Forest & KNN
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "random_forest_model.pkl")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
joblib.dump(knn, "knn_model.pkl")

print("✅ Models trained and saved.")

# -----------------------------
# 5. Load models and evaluate on new data
# -----------------------------
new_data_path = "../Data/V24-2025__0009.parquet"  # replace with actual new dataset
new_df = pd.read_parquet(new_data_path).fillna(0)

X_new = new_df.drop(columns=[target_column]).values
y_new = new_df[target_column].values

scaler = joblib.load("scaler.pkl")
X_new = scaler.transform(X_new)

# Reload models
mlp_loaded = MLP(input_dim, hidden_dim=64, output_dim=num_classes)
mlp_loaded.load_state_dict(torch.load("mlp_model.pth"))
mlp_loaded.eval()

rf_loaded = joblib.load("random_forest_model.pkl")
knn_loaded = joblib.load("knn_model.pkl")

# Predict
with torch.no_grad():
    y_pred_mlp = torch.argmax(mlp_loaded(torch.tensor(X_new, dtype=torch.float32)), axis=1).numpy()

y_pred_rf = rf_loaded.predict(X_new)
y_pred_knn = knn_loaded.predict(X_new)

# -----------------------------
# 6. Evaluate and compare
# -----------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 {name} Metrics:")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model("MLP", y_new, y_pred_mlp)
evaluate_model("Random Forest", y_new, y_pred_rf)
evaluate_model("KNN", y_new, y_pred_knn)
