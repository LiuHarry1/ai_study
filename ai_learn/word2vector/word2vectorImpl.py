import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Prepare the data
texts = ["I like to play football", "I love pizza", "Football is my favorite sport"]

# Tokenize the texts and create sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Create word-to-index mapping
word_index = tokenizer.word_index

# Set parameters
vocab_size = len(word_index) + 1
embedding_dim = 100
window_size = 2

# Generate training data
data = []
for seq in sequences:
    for i in range(len(seq)):
        context_word = seq[i]
        for j in range(i - window_size, i + window_size + 1):
            if j != i and j >= 0 and j < len(seq):
                data.append([context_word, seq[j]])

# Convert training data to numpy arrays
data = np.array(data)
x = data[:, 0]
y = to_categorical(data[:, 1], num_classes=vocab_size)

# Define the Word2Vec model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=1))
model.add(Lambda(lambda x: x[:, 0, :]))
model.add(Dense(vocab_size, activation='softmax'))

model.summary()

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam')
early_stopping = EarlyStopping(monitor='loss', patience=5)
model.fit(x, y, epochs=100, batch_size=64, callbacks=[early_stopping])

# Get the learned word embeddings
embeddings = model.layers[0].get_weights()[0]

# Print the word embeddings
for word, i in word_index.items():
    print(word, embeddings[i])
