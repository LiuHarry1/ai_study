import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Create a BatchNormalization layer
batch_norm_layer = BatchNormalization( epsilon=0.5, momentum=0, scale=False, center=False, moving_variance_initializer="zeros",
                                       beta_initializer="zeros", gamma_initializer="zeros")

# Sample input data with shape (batch_size, num_features)
sample_input = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 2]])

# Print the input data
print("Input data:")
print(sample_input, sample_input.shape)

# Calculate the mean and variance of the input data
mean = np.mean(sample_input, axis=0)
variance = np.var(sample_input, axis=0)

print("mean:", mean, ", variance :" , variance)

# Perform BatchNormalization manually
epsilon = 0.5  # Small value to prevent division by zero
normalized_output = (sample_input - mean) / np.sqrt(variance + epsilon)

# Print the output of manual BatchNormalization
print("Output after manual BatchNormalization:")
print(normalized_output)

# Use the BatchNormalization layer to process the input
tf_output = batch_norm_layer(sample_input)

print(batch_norm_layer.get_weights())

# Print the output using BatchNormalization layer
print("Output after BatchNormalization layer:")
print(tf_output)
