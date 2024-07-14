from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, Bidirectional
import numpy as np

def simple_lstm_layer():
    # Create a dense layer with 10 output neurons and input shape of (None, 20)
    model = Sequential()
    model.add(Bidirectional(LSTM(3, return_sequences=True), input_shape=(3, 2)))
    model.add(Dense(1))  # Output layer with one neuron
    print(model.summary())



def change_weight():
    # Create a simple Bidirectional LSTM layer
    lstm_layer = LSTM(units=3, input_shape=(3, 2), activation=None, recurrent_activation=None, return_sequences=True,
                      return_state= True)

    bi_lstm_layer = Bidirectional(lstm_layer, merge_mode='concat')

    # Simulate input data (batch size of 1 for demonstration)
    input_data = np.array([
                [[1.0, 2], [2, 3], [3, 4]],
                [[5, 6], [6, 7], [7, 8]],
                [[9, 10], [10, 11], [11, 12]]
        ])

    # Pass the input data through the layer to initialize the weights and biases
    bi_lstm_layer(input_data)

    kernel, recurrent_kernel, biases, backward_kernel, backward_recurrent_kernel, backward_biases = bi_lstm_layer.get_weights()

    # Print the initial weights and biases
    print("forward recurrent_kernel:", recurrent_kernel, recurrent_kernel.shape ) # (3,12)
    print('forward kernal:',kernel, kernel.shape) #(2,12)
    print('forward biase: ',biases , biases.shape) # (12)

    print("backward recurrent_kernel:", backward_kernel, backward_kernel.shape ) # (3,12)
    print('backward kernal:',backward_recurrent_kernel, backward_recurrent_kernel.shape) #(2,12)
    print('backward biase: ',backward_biases , backward_biases.shape) # (12)


    kernel = np.array([[2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                       [1, 1, 0, 1, 1, 0, 0, 1, 1 ,0, 0, 0],])

    recurrent_kernel = np.array([[1, 0, 0, 1, 2,1,0,1,2,0,1,0],
                                 [1, 1, 0, 0, 2,1,0,1,2,2,0,0],
                                 [1, 0, 1, 2, 0,1,0,1,1,0,1,0]])

    biases = np.array([3, 1, 0, 1, 1,0,0,1,0,2,0.0,0])

    bi_lstm_layer.set_weights([kernel, recurrent_kernel, biases, kernel, recurrent_kernel, biases])
    print(bi_lstm_layer.get_weights())

    test_data = np.array([
        [[1,0.0]]
    ])

    output, memory_state, carry_state, backward_memory_state, backward_carry_state  = bi_lstm_layer(test_data)

    print('output = ',output.numpy())
    print('forward memory_state = ', memory_state.numpy())
    print('forward carry_state = ',carry_state.numpy())
    print('backward memory state = ', backward_memory_state.numpy())
    print('backward carry state = ',backward_carry_state.numpy())

if __name__ == '__main__':
    # simple_lstm_layer()
    change_weight()

    # https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm