import os
import numpy as np
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import tensorflow as tf
from keras import layers, models

# Constants
MAX_LEN = 200
VOCAB_SIZE = 30000
TRAIN_DIR = '/Users/harry/Documents/apps/ml/aclImdb/train'
TEST_DIR = '/Users/harry/Documents/apps/ml/aclImdb/test'

def read_imdb_data(directory):
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(1 if label_type == 'pos' else 0)  # 1 for pos, 0 for neg
    return texts, labels

def get_dataset():
    # Load the data
    train_texts, train_labels = read_imdb_data(TRAIN_DIR)
    test_texts, test_labels = read_imdb_data(TEST_DIR)

    # Merge and split dataset for training and validation
    texts = train_texts + test_texts
    labels = train_labels + test_labels
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"Train set size: {len(train_texts)}, Test set size: {len(test_texts)}")
    return train_texts, train_labels, test_texts, test_labels

def tokenize_texts(texts, tokenizer, max_length=MAX_LEN):
    tokenized_texts = [tokenizer.encode(text.lower()).ids for text in texts]
    # Padding/truncating to fixed length
    return np.array([t[:max_length] if len(t) > max_length else np.pad(t, (0, max_length - len(t)), 'constant') for t in tokenized_texts])

def train_tokenizer(train_texts):
    # Initialize and train the BPE tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.train_from_iterator(train_texts, trainer)
    tokenizer.save("custom_tokenizer.json")
    return tokenizer

def get_tokenized_dataset(train_texts, train_labels, test_texts, test_labels):
    tokenizer = Tokenizer.from_file("custom_tokenizer.json")
    tokenized_train_texts = tokenize_texts(train_texts, tokenizer)
    tokenized_test_texts = tokenize_texts(test_texts, tokenizer)
    return np.array(tokenized_train_texts), np.array(train_labels), np.array(tokenized_test_texts), np.array(test_labels)

def build_model():
    tokenizer = Tokenizer.from_file("custom_tokenizer.json")
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(input_dim=tokenizer.get_vocab_size(), output_dim=MAX_LEN),
        layers.Conv1D(filters=MAX_LEN, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=MAX_LEN, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(MAX_LEN, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, train_texts, train_labels):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1),
        tf.keras.callbacks.ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir="logs")
    ]
    model.fit(train_texts, train_labels, epochs=20, batch_size=100, validation_split=0.1, callbacks=callbacks)

def evaluate_model(model, test_texts, test_labels):
    test_loss, test_acc = model.evaluate(test_texts, test_labels)
    print(f"Test Accuracy: {test_acc}")

    # Example predictions

    new_texts = ["This movie was amazing!",
                 "I did not like this movie at all.",
                 "I didn't like this movie at all.",
                 "this is bad movie ",
                 "This is good movie",
                 "This isn't good movie",
                 "This is not good movie",
                 "I don't like this movie at all",
                 "i think this is bad movie"]
    tokenizer = Tokenizer.from_file("custom_tokenizer.json")
    tokenized_new_texts = tokenize_texts(new_texts, tokenizer)
    predictions = model.predict(tokenized_new_texts)
    print(predictions)  # Output probabilities

if __name__ == '__main__':
    train_texts, train_labels, test_texts, test_labels = get_dataset()
    train_tokenizer(train_texts)
    train_texts, train_labels, test_texts, test_labels = get_tokenized_dataset(train_texts, train_labels, test_texts, test_labels)
    model = build_model()
    train_model(model, train_texts, train_labels)
    evaluate_model(model, test_texts, test_labels)
