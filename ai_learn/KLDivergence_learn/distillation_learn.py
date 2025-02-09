import tensorflow as tf
import keras
from keras import layers

# 构建一个简单的教师模型
teacher_model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(10, activation="softmax")
])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0
y_train, y_test = keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)

teacher_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
teacher_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

temperature = 5.0
def soft_targets(logits):
    return tf.nn.softmax(logits / temperature)

y_teacher = soft_targets(teacher_model.predict(x_train))

student_model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(784,)),
    layers.Dense(10, activation="softmax")
])

student_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.KLDivergence(),  # 使用 KL 散度
    metrics=["accuracy"]
)

student_model.fit(x_train, y_teacher, epochs=5, batch_size=32, validation_data=(x_test, y_test))