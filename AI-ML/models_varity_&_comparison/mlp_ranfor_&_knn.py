import pandas as pd
import numpy as np
import joblib
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error
)
#
# # =========================================================
# # 1. Load and preprocess data
# # =========================================================
# def load_parquet_dataset(file_path, target_column):
#     df = pd.read_parquet(file_path).fillna(0)
#     X = df.drop(columns=[target_column]).values
#     y = df[target_column].values
#     return X, y
#
# file_path = "../Data/V24-2025__0001.parquet"
target_column = "TempSaug [°C]"  # continuous target

# X, y = load_parquet_dataset(file_path, target_column)
#
# # Optional scaling (important for KNN and MLP)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # =========================================================
# # 2. Define MLP Regressor
# # =========================================================
# # class MLPRegressor(nn.Module):
# #     def __init__(self, input_dim, hidden_dim=64, output_dim=1):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(input_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, output_dim)
# #         )
# #
# #     def forward(self, x):
# #         return self.net(x)
#
# # =========================================================
# # 3. Train MLP
# # =========================================================
# # input_dim = X_train.shape[1]
# # mlp = MLPRegressor(input_dim=input_dim, hidden_dim=64, output_dim=1)
# # print(mlp)
# #
# # criterion = nn.MSELoss()
# # optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
# #
# # train_loader = DataLoader(TensorDataset(
# #     torch.tensor(X_train, dtype=torch.float32),
# #     torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
# # ), batch_size=32, shuffle=True)
# #
# # for epoch in range(15):
# #     total_loss = 0
# #     for inputs, targets in train_loader:
# #         optimizer.zero_grad()
# #         outputs = mlp(inputs)
# #         loss = criterion(outputs, targets)
# #         loss.backward()
# #         optimizer.step()
# #         total_loss += loss.item()
# #     print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")
# #
# # # Save MLP model and scaler
# # torch.save(mlp.state_dict(), "mlp_regressor.pth")
# # joblib.dump(scaler, "scaler.pkl")
#
# # =========================================================
# # 4. Train Random Forest & KNN Regressors
# # =========================================================
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# joblib.dump(rf, "random_forest_regressor.pkl")
#
# knn = KNeighborsRegressor(n_neighbors=5)
# knn.fit(X_train, y_train)
# joblib.dump(knn, "knn_regressor.pkl")
#
# print("✅ Models trained and saved.")

# =========================================================
# 5. Load models and evaluate on new data
# =========================================================
new_data_path = "../Data/V24-2025__0009.parquet"  # replace with actual dataset
new_df = pd.read_parquet(new_data_path).fillna(0)

X_new = new_df.drop(columns=[target_column]).values
y_new = new_df[target_column].values

scaler = joblib.load("scaler.pkl")
X_new = scaler.transform(X_new)

# Reload MLP
# mlp_loaded = MLPRegressor(input_dim, hidden_dim=64, output_dim=1)
# mlp_loaded.load_state_dict(torch.load("mlp_regressor.pth"))
# mlp_loaded.eval()

rf_loaded = joblib.load("random_forest_regressor.pkl")
knn_loaded = joblib.load("knn_regressor.pkl")

# Predict
# with torch.no_grad():
#     y_pred_mlp = mlp_loaded(torch.tensor(X_new, dtype=torch.float32)).squeeze().numpy()

y_pred_rf = rf_loaded.predict(X_new)
y_pred_knn = knn_loaded.predict(X_new)

# =========================================================
# 6. Evaluate and compare
# =========================================================
def evaluate_regressor(name, y_true, y_pred):
    print(f"\n📈 {name} Regression Metrics:")
    print(f"R² Score   : {r2_score(y_true, y_pred):.4f}")
    print(f"MAE        : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE        : {mean_squared_error(y_true, y_pred):.4f}")

# evaluate_regressor("MLP Regressor", y_new, y_pred_mlp)
evaluate_regressor("Random Forest Regressor", y_new, y_pred_rf)
evaluate_regressor("KNN Regressor", y_new, y_pred_knn)


# =========================================================
# 7. Feature Importance Plot (Random Forest)
# =========================================================
import matplotlib.pyplot as plt
import numpy as np

# Check if model supports feature importances
if hasattr(rf_loaded, "feature_importances_"):
    importances = rf_loaded.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("🌲 Random Forest Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Feature importances not available for this model.")


# =========================================================
# 8. Compare Random Forest vs KNN Performance
# =========================================================

# Compute metrics for both models on same new data
rf_r2 = r2_score(y_new, y_pred_rf)
rf_mae = mean_absolute_error(y_new, y_pred_rf)
rf_mse = mean_squared_error(y_new, y_pred_rf)

knn_r2 = r2_score(y_new, y_pred_knn)
knn_mae = mean_absolute_error(y_new, y_pred_knn)
knn_mse = mean_squared_error(y_new, y_pred_knn)

# Print numeric comparison
print("\n📊 Model Comparison (on New Data):")
print(f"{'Metric':<10} | {'Random Forest':>15} | {'KNN':>15}")
print("-" * 45)
print(f"{'R²':<10} | {rf_r2:>15.4f} | {knn_r2:>15.4f}")
print(f"{'MAE':<10} | {rf_mae:>15.4f} | {knn_mae:>15.4f}")
print(f"{'MSE':<10} | {rf_mse:>15.4f} | {knn_mse:>15.4f}")

# Bar chart for visual comparison
metrics = ['R² Score', 'MAE', 'MSE']
rf_values = [rf_r2, rf_mae, rf_mse]
knn_values = [knn_r2, knn_mae, knn_mse]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, rf_values, width, label='Random Forest')
plt.bar(x + width/2, knn_values, width, label='KNN')

plt.title("⚖️ Model Performance Comparison (RF vs KNN)")
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()


# # =========================================================
# # 8. Hyperparameter Tuning (Optional)
# # =========================================================
# from sklearn.model_selection import GridSearchCV
#
# print("\n🔍 Starting GridSearchCV for RandomForestRegressor (this may take a while)...")
#
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# grid_search = GridSearchCV(
#     RandomForestRegressor(random_state=42),
#     param_grid=param_grid,
#     cv=3,
#     scoring='r2',
#     n_jobs=-1,
#     verbose=2
# )
#
# grid_search.fit(X_train, y_train)
#
# print("\n✅ Grid Search Complete!")
# print("Best Parameters:", grid_search.best_params_)
# print("Best Cross-Validation R²:", grid_search.best_score_)
#
# # Refit the model with best params
# best_rf = grid_search.best_estimator_
#
# # Evaluate tuned model on test data
# y_pred_best = best_rf.predict(X_test)
# print("\n🎯 Tuned Random Forest Performance on Test Set:")
# print(f"R² Score   : {r2_score(y_test, y_pred_best):.4f}")
# print(f"MAE        : {mean_absolute_error(y_test, y_pred_best):.4f}")
# print(f"MSE        : {mean_squared_error(y_test, y_pred_best):.4f}")
#
# # Optional: plot feature importances for tuned model
# if hasattr(best_rf, "feature_importances_"):
#     importances = best_rf.feature_importances_
#     indices = np.argsort(importances)[::-1]
#
#     plt.figure(figsize=(10, 5))
#     plt.title("🌟 Tuned Random Forest Feature Importances")
#     plt.bar(range(len(importances)), importances[indices])
#     plt.xlabel("Feature Index")
#     plt.ylabel("Importance")
#     plt.tight_layout()
#     plt.show()
#
