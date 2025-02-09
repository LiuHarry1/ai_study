import tensorflow as tf
import keras
from keras import layers, Model, Input
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras.datasets import imdb

from keras.utils import pad_sequences

# 自定义 MOE 层
class MoeLayer(layers.Layer):
    def __init__(self, num_experts, units, top_k=1, **kwargs):
        """
        :param num_experts: 专家数量
        :param units: 每个专家输出的维度
        :param top_k: 在门控中选择的专家数量（若 top_k < num_experts，则仅激活得分最高的 top_k 个专家）
        """
        super(MoeLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.units = units
        self.top_k = top_k

    def build(self, input_shape):
        # 构建多个专家，每个专家为一个简单的全连接层
        self.experts = [Dense(self.units, activation='relu') for _ in range(self.num_experts)]
        # 构建门控网络：输出维度为专家数量，后接 softmax 生成概率分布
        self.gate_dense = Dense(self.num_experts, activation='softmax')
        super(MoeLayer, self).build(input_shape)

    def call(self, inputs):
        # 计算门控网络输出，shape: (batch_size, num_experts)
        gate_scores = self.gate_dense(inputs)

        # 当 top_k 小于专家数量时，仅保留得分最高的 top_k 专家
        if self.top_k < self.num_experts:
            # 对每个样本选取得分最高的 top_k 专家
            top_k_values, top_k_indices = tf.math.top_k(gate_scores, k=self.top_k)
            # 构造 mask：仅保留 top_k 专家，其它位置置 0
            mask = tf.one_hot(top_k_indices, depth=self.num_experts)
            mask = tf.reduce_sum(mask, axis=1)  # shape: (batch_size, num_experts)
            mask = tf.cast(mask, gate_scores.dtype)
            # 对 gate_scores 应用 mask，并重新归一化
            gate_scores = gate_scores * mask
            gate_scores = gate_scores / tf.reduce_sum(gate_scores, axis=-1, keepdims=True)

        # 分别计算每个专家的输出，得到列表，每个元素 shape: (batch_size, units)
        expert_outputs = [expert(inputs) for expert in self.experts]
        # 将所有专家输出堆叠，shape: (batch_size, num_experts, units)
        expert_outputs = tf.stack(expert_outputs, axis=1)
        # 将 gate_scores 扩展维度，shape: (batch_size, num_experts, 1)
        gate_scores = tf.expand_dims(gate_scores, axis=-1)
        # 对各专家输出进行加权求和，得到最终输出：shape (batch_size, units)
        output = tf.reduce_sum(expert_outputs * gate_scores, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

# 构建文本分类模型
def create_text_classification_model(vocab_size, embedding_dim, max_len, num_classes=2):
    inputs = Input(shape=(max_len,))
    # 嵌入层，将单词索引映射为稠密向量
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
    # 全局平均池化：对整个序列进行汇聚
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    # 插入 MOE 层：例如使用 4 个专家，激活 top-2 专家，每个专家输出 64 维
    x = MoeLayer(num_experts=4, units=64, top_k=2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

if __name__ == '__main__':
    # 设置参数
    vocab_size = 10000   # 词汇表大小
    max_len = 200        # 每个评论截断或补齐到 200 个单词
    embedding_dim = 128  # 词向量维度

    # 加载 IMDB 数据集（正面/负面情感二分类）
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    # 对评论序列进行填充，使所有序列长度一致
    x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')

    # 构建并编译模型
    model = create_text_classification_model(vocab_size, embedding_dim, max_len, num_classes=2)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # 训练模型（这里只训练 3 个 epochs 以示例演示为主）
    model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

    # 在测试集上评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
