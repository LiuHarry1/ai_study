import tensorflow as tf
import numpy as np

# 定义 KLDivergence 损失函数
kl_loss = tf.keras.losses.KLDivergence()

# 真实分布 p (标签)
p = np.array([0.1, 0.4, 0.5], dtype=np.float32)

# 预测分布 q
q = np.array([0.2, 0.3, 0.5], dtype=np.float32)

# 计算 KL 散度损失
loss_value = kl_loss(p, q)

print(f'KL Divergence Loss: {loss_value.numpy()}')
