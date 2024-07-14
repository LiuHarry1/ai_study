import keras.optimizers.optimizer
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

def simple_rnn_layer():

    # Create a dense layer with 10 output neurons and input shape of (None, 20)
    model = Sequential()
    model.add(SimpleRNN(units=3, input_shape=(3, 2),))  # 4 units in the RNN layer, input_shape=(timesteps, features)
    model.add(Dense(1))  # Output layer with one neuron

    # Print the summary of the dense layer
    print(model.summary())

def change_weight():
    # Create a simple Dense layer
    rnn_layer = SimpleRNN(units=3, input_shape=(3, 2), activation=None, return_sequences=True, return_state=True)

    # Simulate input data (batch size of 1 for demonstration)
    input_data = np.array([
                [[1.0, 2], [2, 3], [3, 4]],
                [[5, 6], [6, 7], [7, 8]],
                [[9, 10], [10, 11], [11, 12]]
        ])

    # Pass the input data through the layer to initialize the weights and biases
    rnn_layer(input_data)

    # Access the weights and biases of the dense layer
    kernel, recurrent_kernel, biases = rnn_layer.get_weights()

    # Print the initial weights and biases
    print("recurrent_kernel:", recurrent_kernel) # (3,3)
    print('kernal:',kernel) #(2,3)
    print('biase: ',biases) # (3)

    kernel = np.array([[1, 0, 2], [2, 1, 3]])
    recurrent_kernel = np.array([[1, 2, 1.0], [1, 0, 1], [0, 1, 0]])
    biases = np.array([0, 0, 1.0])

    rnn_layer.set_weights([kernel, recurrent_kernel, biases])
    print(rnn_layer.get_weights())

    test_data = np.array([
        [[1.0, 3], [1, 1], [2, 3]]
    ])

    output, new_state = rnn_layer(test_data)

    print(output)
    print(new_state)


def train_model():
    # Sample sequential data
    # Each sequence has three timesteps, and each timestep has two features
    data = np.array([
        [[1, 2], [2, 3], [3, 4]],
        [[5, 6], [6, 7], [7, 8]],
        [[9, 10], [10, 11], [11, 12]]
    ])
    test_data = np.array([
        [[13, 14], [15, 16], [17, 18]],
        [[19, 20], [21, 22], [23, 24]],
        [[25, 26], [27, 28], [29, 30]]
    ])
    print('data.shape= ', data.shape)
    # Define the RNN model
    model = Sequential()
    model.add(SimpleRNN(units=4, input_shape=(3, 2),
                        name="simpleRNN"))  # 4 units in the RNN layer, input_shape=(timesteps, features)
    model.add(Dense(1, name="output"))  # Output layer with one neuron
    # Compile the model
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01))
    # Print the model summary
    model.summary()
    before_RNN_weight = model.get_layer("simpleRNN").get_weights()
    print('before train ', before_RNN_weight)
    # Train the model
    model.fit(data, np.array([[10], [20], [30]]), epochs=2000, verbose=1)
    RNN_weight = model.get_layer("simpleRNN").get_weights()
    print('after train ', len(RNN_weight), )
    for i in range(len(RNN_weight)):
        print('====', RNN_weight[i].shape, RNN_weight[i])
    # Make predictions
    predictions = model.predict(data)
    print("Predictions:", predictions.flatten())
    predictions = model.predict(test_data)
    print("test data Predictions:", predictions.flatten())

if __name__ == '__main__':
    # simple_rnn_layer()
    change_weight()

