import tensorflow as tf
import numpy as np


def softmax(t):
    s_value = np.exp(t) / np.sum(np.exp(t), axis=-1, keepdims=True)
    # print('softmax value: ', s_value)
    return s_value

def numpy_attention(query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
    query = query[0]
    value = value[0]
    if key is None:
        key = value[0]
    else:
        key = key[0]

    query = np.multiply(query, 1.0 / np.sqrt(float(3)))

    score = np.matmul(query, key.transpose())
    attention_score_np = softmax(score)
    result = np.matmul(attention_score_np, value)
    print('attention score in numpy =', attention_score_np)
    print('result in numpy = ', result)


def test1():
    layer = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)
    query = tf.keras.Input(shape=[8, 16])
    value = tf.keras.Input(shape=[4, 16])
    key = tf.keras.Input(shape=[4, 16])
    attention_output, attention_scores = layer(query, value, key,
                                               return_attention_scores=True)
    print(attention_output.shape)

    print(attention_scores.shape)

def verify_logic_in_multiattention_with_query_key_value():
    query_data = np.array(
       [ [[1, 0.0, 1],[2, 3, 1]]]
    )
    value_data = np.array(
       [ [[2, 1.0, 1],[1, 4, 2 ]]]
    )
    key_data = np.array(
        [[[1, 2.0, 2], [3, 1, 0.1]]]
    )
    print(query_data.shape)

    numpy_attention(query_data, value_data, key_data, return_attention_scores=True)
    print("=============following is keras attention output================")

    multiHeadAttention_layer= tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=2)



    result, attention_scores = multiHeadAttention_layer(query_data, value_data, key_data, return_attention_scores=True)

    query_kernal = np.array([[[1, 1.0]], [[1, 1.0]], [[1, 1.0]]])
    query_biase = np.array([[0., 0.]])

    print(query_kernal.shape)
    output_kernal = np.array([[[1.0, 1, 1.0], [1., 1, 1]]])
    output_biase = np.array([0., 0, 0])
    multiHeadAttention_layer.set_weights(
        [query_kernal, query_biase, query_kernal, query_biase, query_kernal, query_biase,
         output_kernal, output_biase])
    result, attention_scores = multiHeadAttention_layer(query_data, value_data, key_data, return_attention_scores=True)

    # print(query_kernal, query_biase, value_kernal, value_biase, key_kernal, key_biase, ouput1, output2)
    print('attention_scores = ', attention_scores)
    print('result=', result);

def case2():
    import tensorflow as tf

    # Define the input data
    batch_size = 1
    seq_length = 2
    embedding_dim = 3

    # Create sample queries, keys, and values
    queries = tf.random.normal(shape=(batch_size, seq_length, embedding_dim))
    keys = tf.random.normal(shape=(batch_size, seq_length, embedding_dim))
    values = tf.random.normal(shape=(batch_size, seq_length, embedding_dim))

    queries = np.array(
        [[[1, 0.0, 1], [2, 3, 1]]]
    )
    keys = np.array(
        [[[2, 1.0, 1], [1, 4, 2]]]
    )
    values = np.array(
        [[[1, 2.0, 2], [3, 1, 0.1]]]
    )

    # Initialize the MultiHeadAttention layer
    num_heads = 1
    multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=2,  return_attention_scores=True)

    # Call the MultiHeadAttention layer with queries, keys, and values
    attention_output = multi_head_attention(queries, keys, values)

    print("Queries shape:", queries.shape)
    print("Keys shape:", keys.shape)
    print("Values shape:", values.shape)
    print("Attention output shape:", attention_output.shape)

if __name__ == '__main__':
    verify_logic_in_multiattention_with_query_key_value()
    # case2()