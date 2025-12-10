import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb

# Code set 1
df = pd.read_csv("../Data/accelerometer.csv")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print(df.head())
print(df.info())
print(df.describe())

# Code set 2
print(df['wconfid'].value_counts())
sns.countplot(data=df, x="wconfid")
plt.title("Distribution of wconfid")
# plt.show()

print(df['pctid'].value_counts())
sns.countplot(data=df, x="pctid")
plt.title("Distribution of pctid")
# plt.show()
#
def acceleration(s):
    return np.round((s.loc['x']**2 + s.loc['y']**2 + s.loc['z']**2)**0.5, 4)

df['accelaration'] = df.apply(acceleration, axis=1)
print(df.head())

sns.scatterplot(data=df, x='wconfid', y='pctid', hue='accelaration')
plt.title("Acceleration by wconfid and pctid")
# plt.show()

# Note: Pairplot can take time with large datasets
# sns.pairplot(df, diag_kind='kde')
# plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='Greens')
plt.title("Correlation Heatmap")
# plt.show()

# Data Processing
X = df.drop(['accelaration'], axis=1)
y = df['accelaration']

# Train, test and cross validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=101)

print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("y_test shape: " + str(y_test.shape))
print("X_validation shape: " + str(X_validation.shape))
print("y_validation shape: " + str(y_validation.shape))

# Model
xg_reg = xgb.XGBRegressor(colsample_bytree=0.8, learning_rate=0.1, max_depth=6, n_estimators=1000, verbosity=1)
xg_reg.fit(X_train, y_train)

# Predictions
y_pred = xg_reg.predict(X_test)
y_pred_validation = xg_reg.predict(X_validation)

# Evaluation Metrics
print("\n=== Test Set Results ===")
print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred):.4f}")
print(f"R² Score: {metrics.r2_score(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.4f}")

# VISUALIZATION OF PREDICTIONS

# 1. Actual vs Predicted Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Acceleration')
plt.ylabel('Predicted Acceleration')
plt.title('Actual vs Predicted Acceleration (Test Set)')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Acceleration')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.show()

# 3. Distribution of Residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)
plt.show()

# 4. Prediction Error Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(y_test)), y_test.values, label='Actual', alpha=0.7)
plt.plot(range(len(y_pred)), y_pred, label='Predicted', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Acceleration')
plt.title('Actual vs Predicted Values (First 100 samples)')
plt.legend()
plt.xlim(0, 100)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
error_percentage = np.abs((y_test - y_pred) / y_test * 100)
plt.hist(error_percentage, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Absolute Percentage Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xg_reg, max_num_features=10)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# 6. Validation Set Performance
print("\n=== Validation Set Results ===")
print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_validation, y_pred_validation):.4f}")
print(f"R² Score: {metrics.r2_score(y_validation, y_pred_validation):.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_validation, y_pred_validation, alpha=0.5, color='green')
plt.plot([y_validation.min(), y_validation.max()], [y_validation.min(), y_validation.max()], 'r--', lw=2)
plt.xlabel('Actual Acceleration')
plt.ylabel('Predicted Acceleration')
plt.title('Actual vs Predicted Acceleration (Validation Set)')
plt.grid(True, alpha=0.3)
plt.show()