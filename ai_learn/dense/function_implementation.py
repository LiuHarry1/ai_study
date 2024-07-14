import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback

# Custom callback to record weights after each epoch
class RecordWeightsCallback(Callback):
    def __init__(self):
        super(RecordWeightsCallback, self).__init__()
        self.weights = []
        self.biases = []

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_layer("dense_layer").get_weights()[0]
        biases = self.model.get_layer("dense_layer").get_weights()[1]
        self.weights.append(weights.flatten().tolist())
        self.biases.append(biases.tolist())
        print(f"Epoch {epoch + 1} - Weights: {weights.flatten()} - Biases: {biases}")

def visualize_weight_change(record_weights_callback: RecordWeightsCallback ):
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
x_train = np.random.rand(1000, 2)  # 1000 samples with 2 features (x1 and x2)
y_train = 2 * x_train[:, 0] + 3 * x_train[:, 1] + np.random.randn(1000) * 0.1 + 4

# Build the neural network
model = Sequential()
model.add(Dense(1, input_shape=(2,), name='dense_layer'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Create an instance of the callback
record_weights_callback = RecordWeightsCallback()

# Train the model
epochs = 400
model.fit(x_train, y_train, epochs=epochs, callbacks=[record_weights_callback])

visualize_weight_change(record_weights_callback)

# Generate random data for testing
x_test = np.array([[1, 1], [2, 3], [3, 4], [4, 5], [5, 6]])

# Test the model with the new data
y_pred = model.predict(x_test)
print("Predicted outputs:")
print(y_pred.flatten())
print(model.get_layer("dense_layer").get_weights())
