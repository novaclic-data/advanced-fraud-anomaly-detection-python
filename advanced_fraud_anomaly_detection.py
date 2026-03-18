import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# 1. SYNTHETIC BANKING TRANSACTION GENERATION
np.random.seed(42)
n_normal = 5000
n_fraud = 50

# Normal Transactions (Low amount, high frequency)
normal_tx = np.random.normal(loc=100, scale=20, size=(n_normal, 2))
# Fraudulent Transactions (High amount or unusual patterns)
fraud_tx = np.random.uniform(low=250, high=500, size=(n_fraud, 2))

X = np.vstack([normal_tx, fraud_tx])
y_true = np.hstack([np.ones(n_normal), -1 * np.ones(n_fraud)])

# 2. ISOLATION FOREST MODEL (UNSUPERVISED LEARNING)
# We set contamination to 1% to match our simulated fraud rate
model = IsolationForest(contamination=0.01, random_state=42)
y_pred = model.fit_predict(X)

# 3. PERFORMANCE METRIC: ANOMALY RECALL
# Detecting the percentage of true frauds identified by the model
detected_frauds = (y_pred == -1) & (y_true == -1)
recall_score = detected_frauds.sum() / n_fraud
print(f"🚨 Anomaly Detection Recall: {recall_score*100:.2f}% effectiveness")

# 4. FRAUD RADAR VISUALIZATION
plt.figure(figsize=(10, 6))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='lightgrey', label='Normal Transaction', alpha=0.6)
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='crimson', label='Fraud Alert (Anomaly)', edgecolors='black')
plt.title('Banking Fraud Radar: Unsupervised Anomaly Detection', fontsize=14)
plt.xlabel('Transaction Frequency (Standardized)')
plt.ylabel('Transaction Amount (USD)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("advanced_fraud_radar.png")
print("\n✅ Visualization 'advanced_fraud_radar.png' generated successfully.")