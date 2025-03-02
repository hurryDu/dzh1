import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K

# 限制内存使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    # 限制CPU内存
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

# 加载和预处理数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# 降低内存消耗 - 减少训练样本 (可选)
# 如果仍然有内存问题，取消下面两行的注释
# train_size = 30000  # 减少样本数
# X_train, y_train = X_train[:train_size], y_train[:train_size]

# 定义自定义残差块 - 更高效学习
def res_block(x, filters, kernel_size=3, stride=1, use_bias=True, name=None):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    # 如果维度不匹配，调整shortcut
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=use_bias)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


# 构建轻量级但高效的模型
inputs = layers.Input(shape=(32, 32, 3))

# 初始卷积
x = layers.Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

# 残差块 stage 1
x = res_block(x, 16)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.1)(x)

# 残差块 stage 2
x = res_block(x, 32)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.2)(x)

# 残差块 stage 3
x = res_block(x, 64)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

# 输出层
outputs = layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x)

model = tf.keras.Model(inputs, outputs)


# 自定义学习率减少策略
def cosine_decay_with_warmup(epoch):
    warmup_epochs = 3
    total_epochs = 15
    alpha = 0.0  # 最小学习率倍率

    if epoch < warmup_epochs:
        # 预热阶段线性增加
        return (epoch / warmup_epochs) * 0.001
    else:
        # 余弦衰减
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.001 * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(np.pi * progress)))


# 使用初始较高学习率，快速开始学习
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 编译模型
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 回调函数
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

# 训练模型 - 使用较小批次和较少轮次
batch_size = 32
epochs = 15

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f'Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}')

# 类别名称映射 - 英文
class_names = [
    "airplane",  # 0
    "automobile",  # 1
    "bird",  # 2
    "cat",  # 3
    "dog",  # 4
    "horse",  # 5
    "forg",  # 6
    "ship",  # 7
    "truck",  # 8
    "deer"  # 9 (正确的CIFAR-10类别)
]

# 可视化训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')

plt.tight_layout()
plt.show()

# 仅可视化少量预测结果以节省内存
predictions = model.predict(X_test[:9], batch_size=9)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])
    plt.title(f'Pred: {class_names[np.argmax(predictions[i])]}\nTrue: {class_names[np.argmax(y_test[i])]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 清理内存
K.clear_session()