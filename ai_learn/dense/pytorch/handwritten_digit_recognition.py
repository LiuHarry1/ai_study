import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 200
learning_rate = 0.001
num_epochs = 10

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, define the loss function and the optimizer
input_size = 28 * 28
hidden1_size = 512
hidden2_size = 256
num_classes = 10

model = NeuralNet(input_size, hidden1_size, hidden2_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Flatten the images
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 300 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

# Get predictions
def get_predictions(model, loader):
    all_preds = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, 28 * 28).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds = torch.cat((all_preds, predicted.cpu()), dim=0)
    return all_preds

predictions = get_predictions(model, test_loader)

# Plot some examples with their predictions
def plot_predictions(images, true_labels, predicted_labels, num_examples=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_examples):
        plt.subplot(5, 2, i + 1)
        image = images[i].numpy().reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {true_labels[i]}, Pred: {predicted_labels[i]}')
        plt.axis('off')
    plt.show()

# Load some test images to visualize predictions
examples = iter(test_loader)
example_data, example_labels = next(examples)

# Display the first 10 images, true labels, and predicted labels
plot_predictions(example_data, example_labels, predictions[:10], num_examples=10)
