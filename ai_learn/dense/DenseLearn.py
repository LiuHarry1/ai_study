import tensorflow as tf
import keras
import numpy as np
from keras.callbacks import Callback


# The dense layer is mathematically represented as:
#
# output = activation(dot(input, weights) + bias)
#
# Where:
#
# "input" is the input data or the output of the previous layer.
# "weights" is the matrix of weights associated with the connections between the input and output neurons.
# "bias" is the bias vector added to the dot product of input and weights.
# "activation" is the activation function applied element-wise to the output.

def simple_dense_layer():

    # Create a dense layer with 10 output neurons and input shape of (None, 20)
    model = tf.keras.Sequential([
     keras.layers.Dense(units=10, input_shape=(20,), activation = 'relu')
    ]);

    # Print the summary of the dense layer
    print(model.summary())


def change_weight():
    # Create a simple Dense layer
    dense_layer = keras.layers.Dense(units=5, activation='relu', input_shape=(10,))

    # Simulate input data (batch size of 1 for demonstration)
    input_data = tf.ones((1, 10))

    # Pass the input data through the layer to initialize the weights and biases
    _ = dense_layer(input_data)

    # Access the weights and biases of the dense layer
    weights, biases = dense_layer.get_weights()

    # Print the initial weights and biases
    print("Initial Weights:")
    print(weights)
    print("Initial Biases:")
    print(biases)

    # Modify the weights and biases (for demonstration purposes)
    new_weights = tf.ones_like(weights)  # Set all weights to 1
    new_biases = tf.zeros_like(biases)  # Set all biases to 0

    # Set the modified weights and biases back to the dense layer
    dense_layer.set_weights([new_weights, new_biases])

    # Access the weights and biases again after modification
    weights, biases = dense_layer.get_weights()

    # Print the modified weights and biases
    print("Modified Weights:")
    print(weights)
    print("Modified Biases:")
    print(biases)

    input_data = tf.constant([[1,1,3,1,2,1,1,1,1,2]])

    output = dense_layer(input_data)

    print(output)



def multi_Layer_perceptron():
    input_dim = 20
    output_dim = 5

    # Create a simple MLP with 2 hidden dense layers
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=output_dim, activation='softmax')
    ])

    # Print the model summary
    print(model.summary())


def custom_Activation_Function():
    def custom_activation(x):
        # return tf.square(tf.nn.tanh(x))
        return tf.square(x)

        # Create a simple Dense layer

    dense_layer = keras.layers.Dense(units=2, activation=custom_activation, input_shape=(4,))

    weights = tf.ones((4,2))
    biases = tf.ones((2))


    input_data = tf.ones((1, 4))
    _ = dense_layer(input_data)
    dense_layer.set_weights([weights, biases])

    # Print the modified weights and biases
    print("Modified Weights:")
    print(dense_layer.get_weights()[0])
    print("Modified Biases:")
    print(dense_layer.get_weights()[1])

    input_data = tf.constant([[1, 2, 3, 1]])

    output = dense_layer(input_data)

    print(output)

class PrintWeightsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_layer("dense_layer").get_weights()[0]
        biases = self.model.get_layer("dense_layer").get_weights()[1]
        print(f"Epoch {epoch + 1} - Weights: {weights.flatten()} - Biases: {biases}")


def certain_function_implementation():
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense

    # Generate random data for training
    np.random.seed(42)
    x_train = np.random.rand(1000, 2)  # 100 samples with 2 features (x1 and x2)
    y_train = 2 * x_train[:, 0] + 3 * x_train[:, 1] + np.random.randn(1000) * 0.1 + 4

    # Build the neural network
    model = Sequential()
    model.add(Dense(1, input_shape=(2,), name = 'dense_layer'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    epochs = 400
    model.fit(x_train, y_train, epochs=epochs, callbacks=[PrintWeightsCallback()])

    # Generate random data for testing
    x_test = np.array([[1, 1], [2, 3], [3, 4], [4, 5], [5, 6]])

    # Test the model with the new data
    y_pred = model.predict(x_test)
    print("Predicted outputs:")
    print(y_pred.flatten())
    print(model.get_layer("dense_layer").get_weights())


if __name__ == '__main__':
    # simple_dense_layer()
    # change_weight()
    # multi_Layer_perceptron()
    # custom_Activation_Function()
    # certain_function_implement()
    # x_train = np.random((1000, 2))
    certain_function_implementation()

    # y_train = np.multiply(x_train, [2, 3]) + 4