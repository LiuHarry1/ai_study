import tensorflow as tf
from keras import layers, Model, Input
import numpy as np


class DeepSeekMoe(layers.Layer):
    def __init__(self, num_experts, units, top_k=2, loss_coef=0.01, **kwargs):
        """
        :param num_experts: 专家数量
        :param units: 每个专家输出的维度
        :param top_k: 每个样本选择激活的专家数量（top_k 必须小于或等于 num_experts）
        :param loss_coef: 负载均衡辅助损失的权重系数
        """
        super(DeepSeekMoe, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.units = units
        self.top_k = top_k
        self.loss_coef = loss_coef

    def build(self, input_shape):
        # 构建多个专家，每个专家为一个 Dense 层
        self.experts = [layers.Dense(self.units, activation='relu')
                        for _ in range(self.num_experts)]
        # 构建门控网络，输出维度为专家数量，经过 softmax 生成概率分布
        self.gate = layers.Dense(self.num_experts, activation='softmax')
        super(DeepSeekMoe, self).build(input_shape)

    def call(self, inputs):
        tf.print("\n[DeepSeekMoe] Input shape:", tf.shape(inputs))
        # 1. 计算门控网络输出（原始 gate_scores），形状 (batch_size, num_experts)
        gate_scores = self.gate(inputs)
        tf.print("[DeepSeekMoe] Raw gate_scores:", gate_scores, summarize=10)

        # 2. 如果 top_k 小于专家总数，则仅保留得分最高的 top_k 专家
        if self.top_k < self.num_experts:
            # 选取得分最高的 top_k 专家
            topk_values, topk_indices = tf.math.top_k(gate_scores, k=self.top_k)
            tf.print("[DeepSeekMoe] topk_values:", topk_values, summarize=10)
            tf.print("[DeepSeekMoe] topk_indices:", topk_indices, summarize=10)
            # 构造 one-hot 掩码，只有 top_k 位置为 1，其它为 0，形状 (batch_size, num_experts)
            mask = tf.reduce_sum(tf.one_hot(topk_indices, depth=self.num_experts), axis=1)
            mask = tf.cast(mask, gate_scores.dtype)
            tf.print("[DeepSeekMoe] Mask:", mask, summarize=10)

            # 计算负载均衡辅助损失：
            # importance：各专家的平均门控概率（理想状态下各专家数值接近）
            importance = tf.reduce_mean(gate_scores, axis=0)  # shape: (num_experts,)
            # load：各专家实际被选中的平均比例
            load = tf.reduce_mean(mask, axis=0)  # shape: (num_experts,)
            tf.print("[DeepSeekMoe] Importance:", importance, summarize=10)
            tf.print("[DeepSeekMoe] Load:", load, summarize=10)
            # 这里采用归一化后方差的方式作为均衡损失
            importance_loss = tf.reduce_mean((importance / (tf.reduce_mean(importance) + 1e-10) - 1) ** 2)
            load_loss = tf.reduce_mean((load / (tf.reduce_mean(load) + 1e-10) - 1) ** 2)
            balance_loss = importance_loss + load_loss
            tf.print("[DeepSeekMoe] Balance loss:", balance_loss)
            self.add_loss(self.loss_coef * balance_loss)

            # 根据掩码对 gate_scores 进行过滤，并重新归一化（每个样本所有激活的专家之和为 1）
            gate_scores = gate_scores * mask
            gate_scores = gate_scores / tf.reduce_sum(gate_scores, axis=-1, keepdims=True)
            tf.print("[DeepSeekMoe] Gate_scores after masking:", gate_scores, summarize=10)
        else:
            mask = tf.ones_like(gate_scores)

        # 3. 分别计算各个专家的输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            tf.print("[DeepSeekMoe] Expert", i, "output shape:", tf.shape(expert_output))
            expert_outputs.append(expert_output)
        # 堆叠所有专家输出，形状 (batch_size, num_experts, units)
        expert_outputs = tf.stack(expert_outputs, axis=1)
        tf.print("[DeepSeekMoe] Stacked expert outputs shape:", tf.shape(expert_outputs))

        # 4. 扩展 gate_scores 维度后，对各专家输出加权求和
        gate_scores_expanded = tf.expand_dims(gate_scores, axis=-1)  # (batch_size, num_experts, 1)
        tf.print("[DeepSeekMoe] Gate_scores expanded shape:", tf.shape(gate_scores_expanded))
        output = tf.reduce_sum(expert_outputs * gate_scores_expanded, axis=1)
        tf.print("[DeepSeekMoe] Final output shape:", tf.shape(output))
        return output


# 构建一个简单的示例模型，测试 DeepSeekMoe 层
def create_model(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    # 先经过一层 Dense 提取特征
    x = layers.Dense(128, activation='relu')(inputs)
    # 插入 DeepSeekMoe 层：例如使用 4 个专家，top-2 路由，每个专家输出 64 维
    x = DeepSeekMoe(num_experts=4, units=64, top_k=2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


if __name__ == '__main__':
    # 模型参数
    input_dim = 20  # 输入特征维度
    num_classes = 3  # 输出类别数
    model = create_model(input_dim, num_classes)
    model.summary()

    # 构造少量样本数据以调试（例如 3 个样本）
    sample_data = np.random.rand(3, input_dim).astype(np.float32)
    print("\nSample input data:")
    print(sample_data)

    # 前向传播测试，同时会输出调试信息
    predictions = model(sample_data)
    print("\nModel predictions:")
    print(predictions.numpy())
