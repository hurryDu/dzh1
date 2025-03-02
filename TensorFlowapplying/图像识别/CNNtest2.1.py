import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K
from PIL import Image
import os

# 限制内存使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

# 定义类别名称
class_names = [
    "airplane",  # 0
    "automobile",  # 1
    "bird",  # 2
    "cat",  # 3
    "dog",  # 4
    "frog",  # 5
    "horse",  # 6
    "ship",  # 7
    "truck",  # 8
    "deer"  # 9
]


# 定义残差块
def res_block(x, filters, kernel_size=3, stride=1, use_bias=True, name=None):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=use_bias)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


# 构建和训练模型的函数
def build_and_train_model():
    print("Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    print("Building model...")
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

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 回调函数
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    # 训练模型
    print("Training model...")
    batch_size = 32
    epochs = 10  # 减少训练轮次

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

    # 显示训练历史
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

    # 保存模型
    model.save('cifar10_classifier.h5')
    print("Model saved as 'cifar10_classifier.h5'")

    return model


# 处理和预测单张图片的函数
def predict_image(model, image_path):
    try:
        # 加载图片
        img = Image.open(image_path)

        # 调整大小为32x32像素
        img = img.resize((32, 32))

        # 将图片转换为numpy数组并进行预处理
        img_array = np.array(img)

        # 检查图片是否为灰度图像
        if len(img_array.shape) == 2:
            # 将灰度图像转换为RGB
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:
            # 处理带透明通道的图像
            img_array = img_array[:, :, :3]

        # 归一化像素值
        img_array = img_array.astype('float32') / 255.0

        # 添加批次维度
        img_array = np.expand_dims(img_array, axis=0)

        # 进行预测
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100

        # 显示结果
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        plt.show()

        print(f"Predicted class: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.2f}%")

        # 显示所有类别的置信度
        plt.figure(figsize=(10, 5))
        plt.bar(class_names, prediction[0] * 100)
        plt.xlabel('Classes')
        plt.ylabel('Confidence (%)')
        plt.title('Prediction Confidence for Each Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {str(e)}")


# 主函数
def main():
    # 检查是否有保存的模型
    if os.path.exists('cifar10_classifier.h5'):
        print("Loading existing model...")
        model = tf.keras.models.load_model('cifar10_classifier.h5')
    else:
        # 如果没有保存的模型，训练一个新模型
        model = build_and_train_model()

    while True:
        # 询问用户输入
        print("\nOptions:")
        print("1. Classify a new image")
        print("2. Retrain model")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            image_path = input("Enter the path to your image file: ")
            predict_image(model, image_path)
        elif choice == '2':
            model = build_and_train_model()
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")


# 启动程序
if __name__ == "__main__":
    main()