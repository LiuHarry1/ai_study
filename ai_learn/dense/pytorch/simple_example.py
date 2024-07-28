import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Step 1: Create a simple dataset
X = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)  # 3 samples, single feature
y = torch.tensor([[2.0], [4.0], [6.0]])                     # True relationship is y = 2 * x

# Step 2: Define a simple linear model
model = nn.Linear(1, 1)  # One input, one output

# Step 3: Define a simple mean squared error loss
criterion = nn.MSELoss()

# Step 4: Forward pass - compute model output
predictions = model(X)

# Step 5: Compute the loss
loss = criterion(predictions, y)

# Step 7: Print gradients for each parameter
print("Before loss.backward():")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.grad}", param.data)
        print("param",  param)

# Step 6: Backward pass - compute gradients
loss.backward()

# Step 7: Print gradients for each parameter
print("Gradients:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.grad}", param)

# Step 8: Plot the data and initial model prediction
def plot_fit(model, X, y, title=''):
    plt.scatter(X.detach().numpy(), y.numpy(), label='Data', color='blue')
    with torch.no_grad():
        plt.plot(X.numpy(), model(X).numpy(), label='Model Prediction', color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.show()

plot_fit(model, X, y, 'Before Gradient Descent')

# Step 9: Perform a manual update of the model parameters
learning_rate = 0.01
with torch.no_grad():
    for param in model.parameters():
        param -= learning_rate * param.grad

# Plot the model prediction after the update
plot_fit(model, X, y, 'After Gradient Descent Step')

# Step 10: Print updated weights
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} after update: {param}")
