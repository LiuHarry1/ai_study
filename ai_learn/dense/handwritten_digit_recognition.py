from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[0])
print(Y_train.shape)

# Preprocess data
X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

print(Y_train[0])

# Build model
model = Sequential()
model.add(Dense(512, input_shape=(28*28,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, epochs=10, batch_size=200, verbose=1, validation_data=(X_test, Y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')


# Predict on the test set
predictions = model.predict(X_test)

# Plot some examples with their predictions
def plot_predictions(images, true_labels, predicted_labels, num_examples=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_examples):
        plt.subplot(5, 2, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {np.argmax(true_labels[i])}, Pred: {np.argmax(predicted_labels[i])}')
        plt.axis('off')
    plt.show()

# Display the first 10 images, true labels, and predicted labels
plot_predictions(X_test, Y_test, predictions, num_examples=10)

