import tensorflow as tf
from keras import layers, Model, Input
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
import numpy as np


# 自定义 MOE 层（带调试信息）
class MoeLayer(layers.Layer):
    def __init__(self, num_experts, units, top_k=1, **kwargs):
        """
        :param num_experts: 专家数量
        :param units: 每个专家输出的维度
        :param top_k: 当 top_k < num_experts 时，仅激活得分最高的 top_k 个专家
        """
        super(MoeLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.units = units
        self.top_k = top_k

    def build(self, input_shape):
        # 构建多个专家，每个专家为一个 Dense 层
        self.experts = [Dense(self.units, activation='relu') for _ in range(self.num_experts)]
        # 构建门控网络：输出专家数量个数，并通过 softmax 归一化为概率分布
        self.gate_dense = Dense(self.num_experts, activation='softmax')
        super(MoeLayer, self).build(input_shape)

    def call(self, inputs):
        tf.print("\n[MoeLayer] Input shape:", tf.shape(inputs))

        # 1. 计算门控网络输出
        gate_scores = self.gate_dense(inputs)
        tf.print("[MoeLayer] Gate scores before top-k:", gate_scores, summarize=20)

        # 2. 如果设置 top_k 小于专家总数，则只保留得分最高的 top_k 专家
        if self.top_k < self.num_experts:
            top_k_values, top_k_indices = tf.math.top_k(gate_scores, k=self.top_k)
            tf.print("[MoeLayer] Top-k values:", top_k_values, summarize=10)
            tf.print("[MoeLayer] Top-k indices:", top_k_indices, summarize=10)

            # 生成 mask：只有 top_k 专家位置为 1，其余为 0
            mask = tf.one_hot(top_k_indices, depth=self.num_experts)
            mask = tf.reduce_sum(mask, axis=1)  # shape: (batch_size, num_experts)
            mask = tf.cast(mask, gate_scores.dtype)
            tf.print("[MoeLayer] Mask after top-k:", mask, summarize=10)

            # 应用 mask，并重新归一化
            gate_scores = gate_scores * mask
            gate_scores = gate_scores / tf.reduce_sum(gate_scores, axis=-1, keepdims=True)
            tf.print("[MoeLayer] Gate scores after masking & normalization:", gate_scores, summarize=20)
        else:
            tf.print("[MoeLayer] No top-k applied, using full gate scores.")

        # 3. 分别计算各个专家的输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            tf.print(f"[MoeLayer] Expert {i} output shape:", tf.shape(expert_output))
            expert_outputs.append(expert_output)
        # 将所有专家输出堆叠，形状为 (batch_size, num_experts, units)
        expert_outputs = tf.stack(expert_outputs, axis=1)
        tf.print("[MoeLayer] Stacked expert outputs shape:", tf.shape(expert_outputs))

        # 4. 将门控权重扩展后加权求和各专家的输出
        gate_scores_expanded = tf.expand_dims(gate_scores, axis=-1)  # (batch_size, num_experts, 1)
        tf.print("[MoeLayer] Gate scores expanded shape:", tf.shape(gate_scores_expanded))

        output = tf.reduce_sum(expert_outputs * gate_scores_expanded, axis=1)
        tf.print("[MoeLayer] Final output shape:", tf.shape(output))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# 构建一个简单的文本分类模型
def create_text_classification_model(vocab_size, embedding_dim, max_len, num_classes=2):
    inputs = Input(shape=(max_len,))
    # 嵌入层：将词索引映射为词向量
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
    # 全局平均池化：汇聚序列信息
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    # 插入 MOE 层：例如 4 个专家，选择 top-2 专家，每个专家输出 64 维
    x = MoeLayer(num_experts=4, units=64, top_k=2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


# 用少量样本数据测试并调试数据流
if __name__ == '__main__':
    # 模型参数
    vocab_size = 1000  # 词汇表大小
    embedding_dim = 16  # 词向量维度
    max_len = 10  # 序列长度较小，便于调试
    num_classes = 2

    # 创建模型并查看结构
    model = create_text_classification_model(vocab_size, embedding_dim, max_len, num_classes)
    model.summary()

    # 创建少量样本数据：2 个样本，每个样本为一个长度为 max_len 的整数序列
    sample_data = np.random.randint(1, vocab_size, size=(2, max_len))
    print("\nSample input data:")
    print(sample_data)

    # 前向传播，观察调试信息（请确保 TensorFlow 处于 eager 模式，默认 TF2.x 是 eager）
    predictions = model(sample_data)
    print("\nModel predictions:")
    print(predictions.numpy())
