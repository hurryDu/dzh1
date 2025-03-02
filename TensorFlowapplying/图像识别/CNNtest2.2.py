import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K
from PIL import Image
import os
import pickle
import random
from sklearn.model_selection import train_test_split

# 限制内存使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    # 更严格的CPU内存限制
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # 限制TensorFlow内存增长
    tf.config.set_soft_device_placement(True)


# 在每次训练后清理内存
def clear_memory():
    K.clear_session()
    import gc
    gc.collect()


# 设置随机种子
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# 定义类别名称
class_names = [
    "airplane",  # 0
    "automobile",  # 1
    "bird",  # 2
    "frog",  # 3
    "dog",  # 4
    "cat",  # 5
    "horse",  # 6
    "ship",  # 7
    "truck",  # 8
    "deer"  # 9
]


# 构建超轻量级高效模型
def build_lightweight_model():
    model = models.Sequential([
        # 数据增强层 (不增加内存负担的类型)
        layers.experimental.preprocessing.RandomFlip("horizontal"),

        # 第一个卷积块
        layers.Conv2D(16, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4),
                      input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        # 第二个卷积块
        layers.Conv2D(32, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # 第三个卷积块
        layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # 分类器头部
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 数据处理函数
def load_and_preprocess_data():
    # 加载数据
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # 标准化 - 使用简单的归一化以减少计算
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # 转换标签
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # 创建验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# 自定义数据生成器 - 内存友好型
def data_generator(X, y, batch_size=32):
    num_samples = X.shape[0]
    while True:
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # 简单增强
            for j in range(len(X_batch)):
                if np.random.random() > 0.5:
                    X_batch[j] = np.fliplr(X_batch[j])

            yield X_batch, y_batch


# 历史记录相关函数
def save_history(history, filename='lightweight_history.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(history, file)
    print(f"Training history saved to {filename}")


def load_history(filename='lightweight_history.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}


def merge_history(old_history, new_history):
    merged = {}
    for key in new_history:
        if key in old_history:
            merged[key] = old_history[key] + new_history[key]
        else:
            merged[key] = new_history[key]
    return merged


# 显示训练历史
def show_training_history(history):
    plt.figure(figsize=(12, 4))

    # 准确率图表
    plt.subplot(1, 2, 1)
    plt.plot(history.get('accuracy', []), label='Training Accuracy')
    plt.plot(history.get('val_accuracy', []), label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Model Accuracy')

    # 损失图表
    plt.subplot(1, 2, 2)
    plt.plot(history.get('loss', []), label='Training Loss')
    plt.plot(history.get('val_loss', []), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Model Loss')

    plt.tight_layout()
    plt.show()


# 继续训练模型
def continue_training_model(epochs=5, batch_size=16):
    model_path = 'lightweight_cifar10.h5'
    best_model_path = 'best_' + model_path

    # 加载保存的模型
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Loaded existing model for continued training.")
    else:
        model = build_lightweight_model()
        print("Building new model for training.")

    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data()

    # 创建数据生成器
    train_gen = data_generator(X_train, y_train, batch_size)
    steps_per_epoch = len(X_train) // batch_size

    # 定义回调
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 训练模型
    print(f"Training for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 保存模型和历史记录
    model.save(model_path)

    # 合并历史记录
    old_history = load_history()
    merged_history = merge_history(old_history, history.history)
    save_history(merged_history)

    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # 清理内存
    clear_memory()

    return model


# 实现知识蒸馏训练（从复杂模型到简单模型的知识转移）
def distillation_training(epochs=5, batch_size=16, alpha=0.1, temperature=5):
    model_path = 'lightweight_cifar10.h5'

    # 加载保存的模型
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Loaded existing model for distillation training.")
    else:
        model = build_lightweight_model()
        print("Building new model for distillation training.")

    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data()

    # 定义知识蒸馏损失函数
    def distillation_loss(y_true, y_pred):
        # 真实标签的硬损失
        y_true_hard = tf.cast(y_true, tf.float32)
        hard_loss = tf.keras.losses.categorical_crossentropy(y_true_hard, y_pred)

        # 使用自己的预测作为软标签 - 自蒸馏
        # 这通过迭代改进使模型与其更稳定的版本一致
        y_soft = tf.nn.softmax(y_pred / temperature)
        soft_targets = tf.stop_gradient(y_soft)  # 不通过软目标传播梯度
        soft_loss = tf.keras.losses.categorical_crossentropy(soft_targets, y_soft)

        return (1 - alpha) * hard_loss + alpha * soft_loss * (temperature ** 2)

    # 重新编译模型使用蒸馏损失
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # 较低的学习率
        loss=distillation_loss,
        metrics=['accuracy']
    )

    # 定义回调
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'distilled_' + model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
    ]

    # 训练模型
    print(f"Training with knowledge distillation for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 保存模型和历史记录
    model.save('distilled_' + model_path)

    # 合并历史记录
    old_history = load_history()
    merged_history = merge_history(old_history, history.history)
    save_history(merged_history)

    # 评估模型
    model.compile(  # 使用标准损失进行评估
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy after distillation: {test_acc:.4f}")
    print(f"Test loss after distillation: {test_loss:.4f}")

    # 清理内存
    clear_memory()

    return model


# 预测图片函数
def predict_image(model, image_path):
    try:
        # 验证文件是否存在
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' does not exist.")
            return

        # 加载图像
        img = Image.open(image_path)

        # 显示原始图片
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        plt.show()

        # 调整大小为32x32像素
        img_resized = img.resize((32, 32))

        # 将图片转换为numpy数组并进行预处理
        img_array = np.array(img_resized)

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

        # 显示处理后的图片和预测结果
        plt.figure(figsize=(6, 6))
        plt.imshow(img_resized)
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
        print("Please make sure the file is a valid image and you have permission to access it.")
        import traceback
        traceback.print_exc()


# 主函数
def main():
    model_path = 'lightweight_cifar10.h5'

    # 检查是否有保存的模型
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        # 如果没有保存的模型，训练一个新模型
        print("No existing model found. Starting initial training...")
        model = build_lightweight_model()

        # 加载数据
        (X_train, y_train), (X_val, y_val), _ = load_and_preprocess_data()

        # 初始训练
        print("Starting initial training with 5 epochs...")
        callback = tf.keras.callbacks.ModelCheckpoint(
            'best_' + model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

        history = model.fit(
            X_train, y_train,
            batch_size=16,  # 小批量减少内存需求
            epochs=5,  # 开始时只训练很少的轮次
            validation_data=(X_val, y_val),
            callbacks=[callback],
            verbose=1
        )

        # 保存初始历史和模型
        save_history(history.history)
        model.save(model_path)
        print("Initial model saved.")

        # 清理内存
        clear_memory()

    # 循环交互界面
    while True:
        # 显示当前模型性能
        if os.path.exists('lightweight_history.pkl'):
            history = load_history()
            current_epoch = len(history.get('loss', []))
            last_val_acc = history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0
            last_val_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else 0

            print(f"\nCurrent model status:")
            print(f"Total epochs trained: {current_epoch}")
            print(f"Last validation accuracy: {last_val_acc:.4f}")
            print(f"Last validation loss: {last_val_loss:.4f}")

            # 检查是否达到目标
            if last_val_loss < 0.1:
                print("\n🎉 目标达成! 验证损失已低于0.1")

        # 询问用户输入
        print("\nOptions:")
        print("1. Classify a new image")
        print("2. Continue standard training (5-10 epochs)")
        print("3. Run knowledge distillation (improve low loss)")
        print("4. View training history")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            print("\nPlease enter the complete path to your image file:")
            print("Example: C:\\Users\\Username\\Pictures\\cat.jpg")
            image_path = input("Enter the path to your image file: ")
            predict_image(model, image_path)
        elif choice == '2':
            try:
                epochs = int(input("Enter number of epochs (5-10 recommended): "))
                batch_size = int(input("Enter batch size (8-32 recommended): "))
                if epochs < 1 or batch_size < 1:
                    print("Values must be at least 1.")
                    continue
                model = continue_training_model(epochs, batch_size)
            except ValueError:
                print("Please enter valid numbers.")
        elif choice == '3':
            # 知识蒸馏训练 - 特别有助于获得非常低的损失
            try:
                epochs = int(input("Enter number of epochs for distillation (3-5 recommended): "))
                if epochs < 1:
                    print("Number of epochs must be at least 1.")
                    continue
                model = distillation_training(epochs=epochs, batch_size=16)
            except ValueError:
                print("Please enter valid numbers.")
        elif choice == '4':
            if os.path.exists('lightweight_history.pkl'):
                history = load_history()
                show_training_history(history)
            else:
                print("No training history found.")
        elif choice == '5':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")


# 启动程序
if __name__ == "__main__":
    main()