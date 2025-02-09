import tensorflow as tf
import keras
from keras import layers
import numpy as np

# 检查 TensorFlow 版本
print("TensorFlow version:", tf.__version__)


# 基础的 GShard 版 MoE
class MoELayer_GShard(layers.Layer):
    def __init__(self, num_experts=4, top_k=2, d_model=512, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = [keras.Sequential([
            layers.Dense(d_model, activation='relu'),
            layers.Dense(d_model)
        ]) for _ in range(num_experts)]
        self.gate = layers.Dense(num_experts, activation='softmax')

    def call(self, inputs):
        logits = self.gate(inputs)
        top_k_values, top_k_indices = tf.nn.top_k(logits, k=self.top_k)

        batch_size, token_length, _ = inputs.shape
        expert_outputs = tf.zeros_like(inputs)

        for i in range(self.top_k):
            expert_id = tf.gather(top_k_indices, i, axis=-1)
            weight = tf.gather(top_k_values, i, axis=-1)

            expert_output = tf.stack([self.experts[j](inputs) for j in range(self.num_experts)], axis=-1)

            indices = tf.stack([
                tf.tile(tf.range(batch_size)[:, None], [1, token_length]),
                tf.tile(tf.range(token_length)[None, :], [batch_size, 1]),
                expert_id
            ], axis=-1)

            selected_expert_output = tf.gather_nd(expert_output, indices)
            expert_outputs += tf.expand_dims(weight, -1) * selected_expert_output

        return expert_outputs


# 改进的 DeepSeek 风格 MoE（包含共享专家和隔离专家）
class MoELayer_DeepSeek(layers.Layer):
    def __init__(self, num_experts=4, top_k=2, d_model=512, capacity_factor=1.2, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_shared_experts = num_experts // 2  # 一半专家共享
        self.num_isolated_experts = num_experts - self.num_shared_experts  # 另一半专家隔离

        self.shared_experts = [keras.Sequential([
            layers.Dense(d_model, activation='relu'),
            layers.Dense(d_model)
        ]) for _ in range(self.num_shared_experts)]

        self.isolated_experts = [keras.Sequential([
            layers.Dense(d_model, activation='relu'),
            layers.Dense(d_model)
        ]) for _ in range(self.num_isolated_experts)]

        self.gate = layers.Dense(num_experts, activation='softmax')

    def call(self, inputs):
        logits = self.gate(inputs)
        top_k_values, top_k_indices = tf.nn.top_k(logits, k=self.top_k)

        # 计算负载均衡损失
        expert_usage = tf.reduce_mean(tf.one_hot(top_k_indices, depth=self.num_experts), axis=0)
        load_balance_loss = tf.reduce_mean(expert_usage)
        self.add_loss(load_balance_loss)

        # 处理共享专家
        shared_outputs = tf.stack([expert(inputs) for expert in self.shared_experts], axis=1)

        # 处理隔离专家
        isolated_outputs = tf.stack([expert(inputs) for expert in self.isolated_experts], axis=1)

        expert_outputs = tf.zeros_like(inputs)
        for i in range(self.top_k):
            expert_id = tf.gather(top_k_indices, i, axis=-1)
            weight = tf.gather(top_k_values, i, axis=-1)

            is_shared = tf.less(expert_id, self.num_shared_experts)
            selected_expert_output = tf.where(
                is_shared,
                tf.gather(shared_outputs, expert_id, batch_dims=1),
                tf.gather(isolated_outputs, expert_id - self.num_shared_experts, batch_dims=1)
            )
            expert_outputs += tf.expand_dims(weight, -1) * selected_expert_output

        return expert_outputs


# 测试 MoE 层
batch_size = 32
token_length = 5
embedding_size = 512
x = tf.random.normal((batch_size, token_length, embedding_size))
moe_gshard = MoELayer_GShard(d_model=embedding_size)
moe_deepseek = MoELayer_DeepSeek(d_model=embedding_size)

out_gshard = moe_gshard(x)
out_deepseek = moe_deepseek(x)

print("GShard MoE output shape:", out_gshard.shape)
print("DeepSeek MoE output shape:", out_deepseek.shape)