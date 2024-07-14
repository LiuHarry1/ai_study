from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM
import numpy as np

def simple_rnn_layer():

    # Create a dense layer with 10 output neurons and input shape of (None, 20)
    model = Sequential()
    model.add(LSTM(units=3, return_sequences=True, input_shape=(3, 2)))  # 4 units in the RNN layer, input_shape=(timesteps, features)
    model.add(Dense(1))  # Output layer with one neuron

    # Print the summary of the dense layer
    print(model.summary())



def change_weight():
    # Create a simple Dense layer
    lstm_layer = LSTM(units=3, input_shape=(3, 2), activation=None, recurrent_activation=None, return_sequences=True,
                      return_state= True)

    # Simulate input data (batch size of 1 for demonstration)
    input_data = np.array([
                [[1.0, 2], [2, 3], [3, 4]],
                [[5, 6], [6, 7], [7, 8]],
                [[9, 10], [10, 11], [11, 12]]
        ])

    # Pass the input data through the layer to initialize the weights and biases
    lstm_layer(input_data)

    kernel, recurrent_kernel, biases = lstm_layer.get_weights()

    # Print the initial weights and biases
    print("recurrent_kernel:", recurrent_kernel, recurrent_kernel.shape ) # (3,3)
    print('kernal:',kernel, kernel.shape) #(2,3)
    print('biase: ',biases , biases.shape) # (3)


    kernel = np.array([[2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                       [1, 1, 0, 1, 1, 0, 0, 1, 1 ,0, 0, 0],])

    recurrent_kernel = np.array([[1, 0, 0, 1, 2,1,0,1,2,0,1,0],
                                 [1, 1, 0, 0, 2,1,0,1,2,2,0,0],
                                 [1, 0, 1, 2, 0,1,0,1,1,0,1,0]])

    biases = np.array([3, 1, 0, 1, 1,0,0,1,0,2,0.0,0])

    lstm_layer.set_weights([kernel, recurrent_kernel, biases])
    print(lstm_layer.get_weights())

    # test_data = np.array([
    #     [[1.0, 3], [1, 1], [2, 3]]
    # ])

    test_data = np.array([
        [[1,0.0]]
    ])

    output, memory_state, carry_state  = lstm_layer(test_data)

    print(output)
    print(memory_state)
    print(carry_state)

if __name__ == '__main__':
    simple_rnn_layer()
    # change_weight()