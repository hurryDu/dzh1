import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 创建线性回归模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# 编译模型
model.compile(optimizer='sgd', loss='mean_squared_error')

# 训练模型
model.fit(X, y, epochs=100)

# 进行预测
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# 可视化结果
plt.scatter(X, y)
plt.plot(X_new, y_predict, color='red', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with TensorFlow")
plt.show()