import tensorflow as tf
import keras
from keras import layers


class MoELayer_GShard(layers.Layer):
    def __init__(self, num_experts=4, top_k=2, d_model=512, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k

        # 定义专家网络 (FFN)
        self.experts = [keras.Sequential([
            layers.Dense(d_model, activation='relu'),
            layers.Dense(d_model)
        ]) for _ in range(num_experts)]

        # 门控网络 (Gating Network)
        self.gate = layers.Dense(num_experts, activation='softmax')

    def call(self, inputs):
        batch_size, token_length, _ = tf.shape(inputs)

        # 计算门控 logits
        logits = self.gate(inputs)  # (batch_size, token_length, num_experts)

        # 选出 top-k 专家
        top_k_values, top_k_indices = tf.nn.top_k(logits, k=self.top_k)  # (batch_size, token_length, top_k)

        # 计算负载均衡损失（Load Balancing Loss）
        expert_usage = tf.reduce_mean(tf.one_hot(top_k_indices, depth=self.num_experts), axis=0)
        load_balance_loss = tf.reduce_mean(expert_usage)
        self.add_loss(load_balance_loss)

        # 计算所有专家的输出
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts],
                                  axis=2)  # (batch_size, token_length, num_experts, d_model)

        # 构造索引
        batch_indices = tf.range(batch_size)[:, None, None]  # (batch_size, 1, 1)
        token_indices = tf.range(token_length)[None, :, None]  # (1, token_length, 1)

        batch_indices = tf.tile(batch_indices, [1, token_length, self.top_k])  # (batch_size, token_length, top_k)
        token_indices = tf.tile(token_indices, [batch_size, 1, self.top_k])  # (batch_size, token_length, top_k)

        gather_indices = tf.stack([batch_indices, token_indices, top_k_indices],
                                  axis=-1)  # (batch_size, token_length, top_k, 3)

        # 选出 top-k 专家输出
        selected_expert_outputs = tf.gather_nd(expert_outputs,
                                               gather_indices)  # (batch_size, token_length, top_k, d_model)

        # 按权重加权求和
        weighted_expert_outputs = tf.reduce_sum(tf.expand_dims(top_k_values, -1) * selected_expert_outputs,
                                                axis=2)  # (batch_size, token_length, d_model)

        return weighted_expert_outputs


# 测试 GShard MoE 层
batch_size = 32
token_length = 5
embedding_size = 512
x = tf.random.normal((batch_size, token_length, embedding_size))

moe_layer = MoELayer_GShard(num_experts=4, top_k=2, d_model=embedding_size)
out = moe_layer(x)

print("MoE GShard output shape:", out.shape)  # (batch_size, token_length, d_model)
