import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import pickle
import time

# 设置内存增长限制，避免OOM错误
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    # 如果没有GPU，限制CPU内存使用
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)


# 全局配置参数
class Config:
    # 音频参数
    SAMPLE_RATE = 16000
    DURATION = 3  # 秒
    N_MFCC = 40  # MFCC特征数量

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # 模型保存路径
    MODEL_PATH = "emotion_recognition_model.h5"
    HISTORY_PATH = "training_history.pkl"
    ENCODER_PATH = "label_encoder.pkl"

    # 情感类别
    EMOTIONS = ["angry", "happy", "neutral", "sad", "surprised"]


# 特征提取函数
def extract_features(file_path, pad_mode='constant'):
    try:
        # 加载音频文件
        audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)

        # 填充或截断到固定长度
        target_length = Config.SAMPLE_RATE * Config.DURATION
        if len(audio) < target_length:
            audio = librosa.util.pad_center(audio, target_length, mode=pad_mode)
        else:
            audio = audio[:target_length]

        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=Config.N_MFCC)

        # 提取色度特征
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        # 提取梅尔频谱
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)

        # 提取对数功率谱
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        # 提取音调特征
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)

        # 组合特征
        features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ))

        return features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None


# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # 将输入重塑为二维数据以用于CNN
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

        # 第一个卷积块
        layers.Conv1D(64, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2, strides=2),
        layers.Dropout(0.2),

        # 第二个卷积块
        layers.Conv1D(128, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2, strides=2),
        layers.Dropout(0.2),

        # 第三个卷积块
        layers.Conv1D(256, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),

        # 分类器头部
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 构建LSTM模型
def build_lstm_model(input_shape, num_classes):
    model = models.Sequential([
        # 将输入重塑为时间序列以用于LSTM
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

        # LSTM层
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.LSTM(256),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # 分类器头部
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 数据加载和预处理
def load_data(data_path):
    print("Loading and processing data...")
    features = []
    labels = []

    # 遍历情感文件夹
    for emotion in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion)

        # 确保这是一个目录
        if not os.path.isdir(emotion_path):
            continue

        # 检查这个情感是否在我们的目标情感列表中
        if emotion.lower() not in [e.lower() for e in Config.EMOTIONS]:
            continue

        print(f"Processing {emotion} files...")

        # 遍历该情感下的所有音频文件
        emotion_files = [f for f in os.listdir(emotion_path) if f.endswith(('.wav', '.mp3', '.ogg'))]

        for audio_file in tqdm(emotion_files):
            file_path = os.path.join(emotion_path, audio_file)

            # 提取特征
            feature = extract_features(file_path)

            if feature is not None:
                features.append(feature)
                labels.append(emotion.lower())

    # 转换为numpy数组
    features = np.array(features)

    # 编码标签
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, categorical_labels, test_size=0.2, random_state=42
    )

    print(f"特征形状: {features.shape}")
    print(f"类别: {encoder.classes_}")
    print(f"训练集: {X_train.shape[0]}个样本, 测试集: {X_test.shape[0]}个样本")

    return X_train, X_test, y_train, y_test, encoder


# 训练模型
def train_model(model_type='cnn'):
    # 输入数据路径
    data_path = input("输入情感语音数据集路径: ")

    if not os.path.exists(data_path):
        print(f"路径不存在: {data_path}")
        return None, None

    # 加载数据
    X_train, X_test, y_train, y_test, encoder = load_data(data_path)

    # 保存标签编码器
    with open(Config.ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)

    # 获取输入维度和类别数量
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]

    # 选择模型类型
    if model_type.lower() == 'cnn':
        model = build_cnn_model(input_shape, num_classes)
        print("使用CNN模型进行训练")
    else:
        model = build_lstm_model(input_shape, num_classes)
        print("使用LSTM模型进行训练")

    # 定义回调函数
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=Config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
    ]

    # 训练模型
    print("\n开始训练...")
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_split=0.2,
        callbacks=callbacks_list,
        verbose=1
    )

    # 保存训练历史
    with open(Config.HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    # 评估模型
    print("\n在测试集上评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试损失: {test_loss:.4f}")

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 混淆矩阵
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 将数字标签转换回情感标签
    emotion_pred = encoder.inverse_transform(y_pred)
    emotion_true = encoder.inverse_transform(y_true)

    # 创建混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    emotion_labels = encoder.classes_
    tick_marks = np.arange(len(emotion_labels))
    plt.xticks(tick_marks, emotion_labels, rotation=45)
    plt.yticks(tick_marks, emotion_labels)

    # 在混淆矩阵中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # 打印分类报告
    report = classification_report(emotion_true, emotion_pred)
    print("\n分类报告:")
    print(report)

    return model, encoder


# 批量预测文件夹中的音频
def batch_predict_emotions(folder_path):
    # 加载模型和编码器
    if not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.ENCODER_PATH):
        print("模型或编码器未找到。请先训练模型。")
        return

    model = tf.keras.models.load_model(Config.MODEL_PATH)

    with open(Config.ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    results = []
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.ogg'))]

    if not audio_files:
        print("没有找到音频文件。")
        return

    print(f"分析文件夹中的 {len(audio_files)} 个音频文件...")

    for audio_file in tqdm(audio_files):
        file_path = os.path.join(folder_path, audio_file)

        # 提取特征
        features = extract_features(file_path)

        if features is None:
            print(f"无法处理文件: {audio_file}")
            continue

        features = np.expand_dims(features, axis=0)  # 添加批次维度

        # 预测情感
        prediction = model.predict(features)[0]
        emotion_idx = np.argmax(prediction)
        emotion = encoder.inverse_transform([emotion_idx])[0]
        confidence = prediction[emotion_idx] * 100

        results.append({
            'file': audio_file,
            'emotion': emotion,
            'confidence': confidence
        })

    # 创建结果表格
    if results:
        df = pd.DataFrame(results)
        print("\n预测结果:")
        print(df)

        # 保存结果
        output_file = "emotion_predictions.csv"
        df.to_csv(output_file, index=False)
        print(f"结果已保存到 {output_file}")

        # 可视化结果
        plt.figure(figsize=(10, 6))
        emotion_counts = df['emotion'].value_counts()
        emotion_counts.plot(kind='bar')
        plt.title('Emotion Distribution in Audio Files')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


# 预测单个音频文件的情感
def predict_emotion(audio_file):
    # 加载模型和编码器
    if not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.ENCODER_PATH):
        print("模型或编码器未找到。请先训练模型。")
        return

    model = tf.keras.models.load_model(Config.MODEL_PATH)

    with open(Config.ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    # 提取特征
    features = extract_features(audio_file)

    if features is None:
        print(f"无法处理文件: {audio_file}")
        return

    features = np.expand_dims(features, axis=0)  # 添加批次维度

    # 预测情感
    prediction = model.predict(features)[0]
    emotion_idx = np.argmax(prediction)
    emotion = encoder.inverse_transform([emotion_idx])[0]
    confidence = prediction[emotion_idx] * 100

    # 显示结果
    print(f"检测到的情感: {emotion} (置信度: {confidence:.2f}%)")

    # 显示所有情感的置信度
    plt.figure(figsize=(10, 5))
    plt.bar(encoder.classes_, prediction * 100)
    plt.xlabel('情感')
    plt.ylabel('置信度 (%)')
    plt.title('情感预测置信度')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 显示音频波形和声谱图
    plt.figure(figsize=(12, 6))

    # 加载音频进行可视化
    audio, sr = librosa.load(audio_file, sr=Config.SAMPLE_RATE)

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f'波形图 - 预测情感: {emotion}')

    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('声谱图')

    plt.tight_layout()
    plt.show()

    return emotion, confidence


# 主函数
def main():
    print("===== 语音情感识别系统 =====")

    while True:
        print("\n选择操作:")
        print("1. 训练新模型 (CNN)")
        print("2. 训练新模型 (LSTM)")
        print("3. 从文件预测情感")
        print("4. 批量分析文件夹中的音频")
        print("5. 退出")

        choice = input("输入选择 (1-5): ")

        if choice == '1':
            train_model('cnn')
        elif choice == '2':
            train_model('lstm')
        elif choice == '3':
            audio_path = input("输入音频文件路径: ")
            if os.path.exists(audio_path):
                predict_emotion(audio_path)
            else:
                print("文件不存在，请检查路径。")
        elif choice == '4':
            folder_path = input("输入音频文件夹路径: ")
            batch_predict_emotions(folder_path)
        elif choice == '5':
            print("退出程序。")
            break
        else:
            print("无效选择，请重新输入。")


if __name__ == "__main__":
    main()