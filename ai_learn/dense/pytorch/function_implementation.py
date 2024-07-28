import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Setting the backend for Matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system

# Custom callback to record weights after each epoch
class RecordWeightsCallback:
    def __init__(self):
        self.weights = []
        self.biases = []

    def on_epoch_end(self, model):
        # Get the weights and biases from the layer
        weights = model.linear.weight.detach().numpy()
        biases = model.linear.bias.detach().numpy()
        self.weights.append(weights.flatten().tolist())
        self.biases.append(biases.tolist())
        print(f"Weights: {weights.flatten()} - Biases: {biases}")

def visualize_weight_change(record_weights_callback: RecordWeightsCallback):
    # Plot the recorded weights
    weights = np.array(record_weights_callback.weights)
    biases = np.array(record_weights_callback.biases)

    plt.figure(figsize=(12, 6))

    # Plot weights
    for i in range(weights.shape[1]):
        plt.plot(weights[:, i], label=f'Weight {i}')

    # Plot biases
    plt.plot(biases, label='Bias', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Weight and Bias Changes Over Epochs')
    plt.legend()
    plt.show(block=True)

# Generate random data for training
np.random.seed(42)
x_train = np.random.rand(1000, 2).astype(np.float32)  # 1000 samples with 2 features (x1 and x2)
y_train = 2 * x_train[:, 0] + 3 * x_train[:, 1] + np.random.randn(1000) * 0.1 + 4
y_train = y_train.astype(np.float32)

# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train).view(-1, 1)

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create an instance of the callback
record_weights_callback = RecordWeightsCallback()

# Train the model
epochs = 400
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Call the custom callback
    record_weights_callback.on_epoch_end(model)

# Visualize weight changes
visualize_weight_change(record_weights_callback)

# Generate random data for testing
x_test = np.array([[1, 1], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=np.float32)
x_test_tensor = torch.from_numpy(x_test)

# Test the model with the new data
model.eval()
with torch.no_grad():
    y_pred = model(x_test_tensor).numpy()

print("Predicted outputs:")
print(y_pred.flatten())

# Print final weights and biases
final_weights = model.linear.weight.detach().numpy()
final_biases = model.linear.bias.detach().numpy()
print("Final Weights and Biases:")
print(final_weights, final_biases)
