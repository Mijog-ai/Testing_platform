import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# -----------------------------
# 1. Load data from Parquet
# -----------------------------
file_path = "../Data/V24-2025__0001.parquet"
target_column = "TempSaug [°C]"  # same as before

df = pd.read_parquet(file_path)
df = df.fillna(0)  # fill missing values

print("✅ Columns:", df.columns.tolist())
print(df.head(5))

# -----------------------------
# 2. Split into features (X) and target (y)
# -----------------------------
X = df.drop(columns=[target_column]).values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Random Forest Regressor
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("\n🌲 Random Forest Results:")
print(f"MSE: {rf_mse:.4f}")
print(f"R²:  {rf_r2:.4f}")

# -----------------------------
# 4. KNN Regressor
# -----------------------------
knn = KNeighborsRegressor(
    n_neighbors=5,
    n_jobs=-1
)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

knn_mse = mean_squared_error(y_test, knn_preds)
knn_r2 = r2_score(y_test, knn_preds)

print("\n🤖 KNN Results:")
print(f"MSE: {knn_mse:.4f}")
print(f"R²:  {knn_r2:.4f}")

print("\n✅ All models trained and evaluated successfully!")
