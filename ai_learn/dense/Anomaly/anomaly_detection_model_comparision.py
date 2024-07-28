import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
df = pd.read_csv(url, header=None)

# Define column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

df.columns = column_names

# Identify normal and attack connections
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# Encode categorical features
categorical_features = ['protocol_type', 'service', 'flag']
for feature in categorical_features:
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])

# Separate features and target variable
X = df.drop(columns=['label'])
y = df['label']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim


# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize the model
input_dim = X_train.shape[1]
model = Autoencoder(input_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert training data (normal connections only) to PyTorch tensors
X_train_normal = X_train[y_train == 0]
X_train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)

# Training loop
num_epochs = 50
batch_size = 256

for epoch in range(num_epochs):
    # Shuffle the data for each epoch
    indices = torch.randperm(X_train_tensor.size(0))
    X_train_tensor_shuffled = X_train_tensor[indices]

    for i in range(0, X_train_tensor.size(0), batch_size):
        batch = X_train_tensor_shuffled[i:i + batch_size]

        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Calculate reconstruction error
with torch.no_grad():
    reconstructed = model(X_test_tensor)
    reconstruction_error = torch.mean((reconstructed - X_test_tensor) ** 2, dim=1)

# Determine the threshold for anomaly detection
threshold = np.percentile(reconstruction_error.numpy(), 95)  # e.g., top 5% as anomalies

# Identify anomalies
anomalies_ae = reconstruction_error.numpy() > threshold
print(f'Number of anomalies detected by Autoencoder: {np.sum(anomalies_ae)}')

# Compare with actual labels
y_pred_ae = np.zeros_like(y_test)
y_pred_ae[anomalies_ae] = 1

# Evaluate the Autoencoder
conf_matrix_ae = confusion_matrix(y_test, y_pred_ae)
print('Confusion Matrix (Autoencoder):')
print(conf_matrix_ae)

report_ae = classification_report(y_test, y_pred_ae, target_names=['Normal', 'Anomaly'])
print('Classification Report (Autoencoder):')
print(report_ae)

roc_auc_ae = roc_auc_score(y_test, reconstruction_error.numpy())
print(f'AUC-ROC (Autoencoder): {roc_auc_ae:.4f}')


from sklearn.ensemble import IsolationForest

# Initialize the Isolation Forest model
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

# Train the Isolation Forest model
isolation_forest.fit(X_train)

# Predict anomalies on the test set
y_pred_if = isolation_forest.predict(X_test)

# Convert Isolation Forest output to binary classification
# -1 means anomaly, 1 means normal
y_pred_if = np.where(y_pred_if == -1, 1, 0)

# Evaluate the Isolation Forest model
conf_matrix_if = confusion_matrix(y_test, y_pred_if)
print('Confusion Matrix (Isolation Forest):')
print(conf_matrix_if)

report_if = classification_report(y_test, y_pred_if, target_names=['Normal', 'Anomaly'])
print('Classification Report (Isolation Forest):')
print(report_if)

roc_auc_if = roc_auc_score(y_test, isolation_forest.decision_function(X_test))
print(f'AUC-ROC (Isolation Forest): {roc_auc_if:.4f}')

# Accuracy
accuracy_ae = accuracy_score(y_test, y_pred_ae)
accuracy_if = accuracy_score(y_test, y_pred_if)

# Precision
precision_ae = precision_score(y_test, y_pred_ae)
precision_if = precision_score(y_test, y_pred_if)

# Recall
recall_ae = recall_score(y_test, y_pred_ae)
recall_if = recall_score(y_test, y_pred_if)

# F1-Score
f1_score_ae = f1_score(y_test, y_pred_ae)
f1_score_if = f1_score(y_test, y_pred_if)

# Display the results
print("\nModel Comparison:")
print(f"{'Metric':<15} {'Autoencoder':<15} {'Isolation Forest':<15}")
print(f"{'Accuracy':<15} {accuracy_ae:<15.4f} {accuracy_if:<15.4f}")
print(f"{'Precision':<15} {precision_ae:<15.4f} {precision_if:<15.4f}")
print(f"{'Recall':<15} {recall_ae:<15.4f} {recall_if:<15.4f}")
print(f"{'F1-Score':<15} {f1_score_ae:<15.4f} {f1_score_if:<15.4f}")
print(f"{'AUC-ROC':<15} {roc_auc_ae:<15.4f} {roc_auc_if:<15.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix for Autoencoder
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_ae, annot=True, fmt="d", cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix (Autoencoder)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Confusion matrix for Isolation Forest
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_if, annot=True, fmt="d", cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix (Isolation Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc

# Autoencoder ROC
fpr_ae, tpr_ae, _ = roc_curve(y_test, reconstruction_error.numpy())
roc_auc_ae = auc(fpr_ae, tpr_ae)

# Isolation Forest ROC
fpr_if, tpr_if, _ = roc_curve(y_test, isolation_forest.decision_function(X_test))
roc_auc_if = auc(fpr_if, tpr_if)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_ae, tpr_ae, color='blue', lw=2, label=f'Autoencoder (AUC = {roc_auc_ae:.4f})')
plt.plot(fpr_if, tpr_if, color='green', lw=2, label=f'Isolation Forest (AUC = {roc_auc_if:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
