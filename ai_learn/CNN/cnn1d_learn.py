
import tensorflow as tf
import numpy as np
import keras



def case1():
    # Input sequence and filter
    input_sequence = np.array([1, 2, 3, 4, 5, 6])
    filter_kernel = np.array([2, -1])

    # Reshape the input sequence and filter to fit Conv1D
    input_sequence = input_sequence.reshape(1, -1, 1)
    filter_kernel = filter_kernel.reshape(-1, 1, 1)

    # Create a Conv1D model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=1, kernel_size=2, activation='linear', use_bias=False,
                               input_shape=(None, 1)),
    ])

    model.summary()

    # Set the weights of the Conv1D layer to the filter_kernel
    model.layers[0].set_weights([filter_kernel])

    # Perform 1D Convolution
    output_sequence = model.predict(input_sequence).flatten()

    print("Input Sequence:", input_sequence.flatten(), "shape:", input_sequence.shape)
    print("Filter:", filter_kernel.flatten(), " shape :",filter_kernel.shape )
    print("Output Sequence:", output_sequence)


def case_custom_activation():
    # Input sequence and filter
    input_sequence = np.array([1, 2, 3, 4, 5, 6])
    filter_kernel = np.array([2, -1])

    # Reshape the input sequence and filter to fit Conv1D
    input_sequence = input_sequence.reshape(1, -1, 1)
    filter_kernel = filter_kernel.reshape(-1, 1, 1)

    def custom_activation(x):
        # return tf.square(tf.nn.tanh(x))
        return tf.square(x)

    # Create a Conv1D model
    model = keras.Sequential([
        keras.layers.Conv1D(filters=1, kernel_size=2, activation=custom_activation, use_bias=False,
                               input_shape=(None, 1)),
    ])

    model.summary()

    # Set the weights of the Conv1D layer to the filter_kernel
    model.layers[0].set_weights([filter_kernel])

    # Perform 1D Convolution
    output_sequence = model.predict(input_sequence).flatten()

    print("Input Sequence:", input_sequence.flatten(), "shape:", input_sequence.shape)
    print("Filter:", filter_kernel.flatten(), " shape :",filter_kernel.shape )
    print("Output Sequence:", output_sequence)



def cnn1d_biase():
    # Input sequence and filter
    input_sequence = np.array([1, 2, 3, 4, 5, 6])
    filter_kernel = np.array([2, -1])
    biase = np.array([2])

    # Reshape the input sequence and filter to fit Conv1D
    input_sequence = input_sequence.reshape(1, -1, 1)
    filter_kernel = filter_kernel.reshape(-1, 1, 1)

    def custom_activation(x):
        # return tf.square(tf.nn.tanh(x))
        return tf.square(x)

    # Create a Conv1D model
    model = keras.Sequential([
        keras.layers.Conv1D(filters=1, kernel_size=2, activation=custom_activation,
                               input_shape=(None, 1)),
    ])

    model.summary()

    print(model.layers[0].get_weights()[0].shape)
    print(model.layers[0].get_weights()[1].shape)

    # Set the weights of the Conv1D layer to the filter_kernel
    model.layers[0].set_weights([filter_kernel, biase])

    # Perform 1D Convolution
    output_sequence = model.predict(input_sequence).flatten()

    print("Input Sequence:", input_sequence.flatten(), "shape:", input_sequence.shape)
    print("Filter:", filter_kernel.flatten(), " shape :", filter_kernel.shape)
    print("Output Sequence:", output_sequence)


if __name__ == '__main__':
    case_custom_activation()
    cnn1d_biase()