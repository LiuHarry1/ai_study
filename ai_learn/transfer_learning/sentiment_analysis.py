import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras as keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

# embedding_url = "https://tfhub.dev/google/nnlm-en-dim50/2"
embedding_url = "/Users/harry/Documents/apps/ml/nnlm_embedding"

index_dic = {"0": "negative", "1": "positive"}


def get_dataset_to_train():
    train_test = np.load('data/train_test.npz', allow_pickle=True)

    x_train = train_test['X_train']
    y_train = train_test['y_train']
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    print(len(X_train), len(X_test))
    return X_train, y_train, X_test, y_test


def get_model():
    hub_layer = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=True)
    # Build the model
    model = Sequential([
        hub_layer,
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model


def train(model, train_data, train_labels, test_data, test_labels):
    # train_data, train_labels, test_data, test_labels = get_dataset_to_train()
    train_data = [tf.compat.as_str(tf.compat.as_bytes(str(x))) for x in train_data]
    test_data = [tf.compat.as_str(tf.compat.as_bytes(str(x))) for x in test_data]

    train_data = np.asarray(train_data)  # Convert to numpy array
    test_data = np.asarray(test_data)  # Convert to numpy array
    print(train_data.shape, test_data.shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=4, mode='max', verbose=1)
    # 定义ModelCheckpoint回调函数
    # checkpoint = ModelCheckpoint( './models/model_new1.h5', monitor='val_sparse_categorical_accuracy', save_best_only=True,
    #                              mode='max', verbose=1)

    checkpoint_pb = ModelCheckpoint(filepath="./models_pb/", monitor='val_sparse_categorical_accuracy',
                                    save_weights_only=False, save_best_only=True)

    history = model.fit(train_data[:2000], train_labels[:2000], epochs=45, batch_size=45,
                        validation_data=(test_data, test_labels), shuffle=True,
                        verbose=1, callbacks=[early_stop, checkpoint_pb])
    print("history", history)

    return model


def evaluate_model(test_data, test_labels):
    model = load_trained_model()
    # Evaluate the model
    results = model.evaluate(test_data, test_labels, verbose=2)

    print("Test accuracy:", results[1])


def predict(real_data):
    model = load_trained_model()
    probabilities = model.predict([real_data]);
    print("probabilities :", probabilities)
    result = get_label(probabilities)
    return result


def get_label(probabilities):
    index = np.argmax(probabilities[0])
    print("index :" + str(index))

    result_str = index_dic.get(str(index))
    # result_str = list(index_dic.keys())[list(index_dic.values()).index(index)]

    return result_str


def load_trained_model():
    # model = get_model()
    # model.load_weights('./models/model_new1.h5')

    model = tf.keras.models.load_model('models_pb')

    return model


def predict_my_module():
    # review = "I don't like it"
    # review = "this is bad movie "
    # review = "This is good movie"
    # review = " this is terrible movie"
    # review = "This isn‘t great movie"
    # review = "i think this is bad movie"
    # review = "I'm not very disappoint for this movie"
    # review = "I'm not very disappoint for this movie"
    # review = "I am very happy for this movie"

    new_texts = ["This movie was amazing!",
                 "I did not like this movie at all.",
                 "I didn't like this movie at all.",
                 "this is bad movie ",
                 "This is good movie",
                 "This isn't good movie",
                 "This is not good movie",
                 "I don't like this movie at all",
                 "i think this is bad movie"]
    s = predict(new_texts)
    print(s)



if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_dataset_to_train()
    model = get_model()
    model = train(model, x_train, y_train, x_test, y_test)
    evaluate_model(x_test, y_test)
    predict_my_module()










