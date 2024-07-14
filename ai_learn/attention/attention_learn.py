import tensorflow as tf
import numpy as np
import keras as keras

def attention_sentence_similarity():
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    value_input = tf.keras.Input(shape=(None,), dtype='int32')

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(value_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    # Add DNN layers, and create Model.
    # ...

    model = keras.Model([query_input, value_input], input_layer)

    print(model.summary())

def softmax(t):
    s_value = np.exp(t) / np.sum(np.exp(t), axis=-1, keepdims=True)
    # print('softmax value: ', s_value)
    return s_value

def numpy_attention(inputs,
        mask=None,
        training=None,
        return_attention_scores=False,
        use_causal_mask=False):

    query = inputs[0]
    value = inputs[1]
    key = inputs[2] if len(inputs) > 2 else value

    score = np.matmul(query, key.transpose())
    attention_score_np = softmax(score)
    result = np.matmul(attention_score_np, value)
    print('attention score in numpy =', attention_score_np)
    print('result in numpy = ', result)


def verify_logic_in_attention_with_query_value():
    query_data = np.array(
        [[1, 0.0, 1],[2, 3, 1]]
    )
    value_data = np.array(
        [[2, 1.0, 1],[1, 4, 2 ]]
    )
    print(query_data.shape)

    numpy_attention([query_data, value_data], return_attention_scores=True)
    print("=============following is keras attention output================")

    attention_layer= tf.keras.layers.Attention()

    result, attention_scores = attention_layer([query_data, value_data], return_attention_scores=True)

    print('attention_scores = ', attention_scores)
    print('result=', result);


def verify_logic_in_attention_with_query_key_value():
    query_data = np.array(
        [[1, 0.0, 1],[2, 3, 1]]
    )
    value_data = np.array(
        [[2, 1.0, 1],[1, 4, 2 ]]
    )
    key_data = np.array(
        [[1, 2.0, 2], [3, 1, 0.1]]
    )
    print(query_data.shape)

    numpy_attention([query_data, value_data, key_data], return_attention_scores=True)
    print("=============following is keras attention output================")

    attention_layer= tf.keras.layers.Attention()

    result, attention_scores = attention_layer([query_data, value_data, key_data], return_attention_scores=True)

    print(attention_layer.get_weights())
    print('attention_scores = ', attention_scores)
    print('result=', result);

def  verify_logic_in_attention_with_mask():
    query_data = np.array(
        [[1, 0.0, 1],[2, 3, 1]]
    )
    value_data = np.array(
        [[2, 1.0, 1],[1, 4, 2 ]]
    )
    key_data = np.array(
        [[1, 2.0, 2], [3, 1, 0.1]]
    )
    print(query_data.shape)

    query_mask = [[1.0, 1]]
    value_mask = [[1.0, 1]]

    masks = [query_mask, value_mask]

    numpy_attention([query_data, value_data, key_data], return_attention_scores=True)
    print("=============following is keras attention output================")

    attention_layer= tf.keras.layers.Attention()

    result, attention_scores = attention_layer([query_data, value_data, key_data], return_attention_scores=True,
                                               mask=masks)

    print(attention_layer.get_weights())

    print('attention_scores = ', attention_scores)
    print('result=', result);

if __name__ == '__main__':
    attention_sentence_similarity()
    # verify_logic_in_attention_with_query_value()
    # verify_logic_in_attention_with_query_key_value()
    # verify_logic_in_attention_with_mask()

# https://keras.io/api/layers/attention_layers/attention/