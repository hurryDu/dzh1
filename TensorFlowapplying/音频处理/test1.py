import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pickle
import wave
import struct


# 全局配置参数
class Config:
    # 音频参数
    SAMPLE_RATE = 16000
    DURATION = 3  # 秒

    # 模型保存路径
    MODEL_PATH = "emotion_recognition_model.pkl"
    SCALER_PATH = "feature_scaler.pkl"
    ENCODER_PATH = "label_encoder.pkl"

    # 情感类别
    EMOTIONS = ["angry", "happy", "neutral", "sad", "surprised"]


# 使用wave库提取特征 (不依赖librosa)
def extract_features_simple(file_path):
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        # 尝试打开wav文件
        try:
            with wave.open(file_path, 'rb') as wf:
                # 获取音频属性
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                n_frames = wf.getnframes()

                # 读取音频数据
                audio_data = wf.readframes(n_frames)

                # 转换为numpy数组
                if sample_width == 2:  # 16-bit音频
                    format_str = f"<{n_frames * channels}h"
                    audio = np.array(struct.unpack(format_str, audio_data))
                elif sample_width == 4:  # 32-bit音频
                    format_str = f"<{n_frames * channels}i"
                    audio = np.array(struct.unpack(format_str, audio_data))
                else:
                    print(f"不支持的样本宽度: {sample_width}")
                    return None

                # 如果是立体声，取平均转为单声道
                if channels == 2:
                    audio = np.mean(audio.reshape(-1, 2), axis=1)
        except Exception as e:
            print(f"打开音频文件时出错: {str(e)}")
            return None

        # 重采样到目标采样率 (简化版，仅截断或补零)
        target_length = int(Config.SAMPLE_RATE * Config.DURATION)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')

        # 提取简单特征 (不依赖librosa)
        # 1. 时域特征
        mean = np.mean(audio)
        std = np.std(audio)
        max_val = np.max(audio)
        min_val = np.min(audio)

        # 2. 分段分析 (简化的频域特征)
        segments = 20
        segment_length = len(audio) // segments
        segment_features = []
        for i in range(segments):
            start = i * segment_length
            end = start + segment_length
            segment = audio[start:end]
            segment_features.extend([
                np.mean(segment),
                np.std(segment),
                np.max(segment),
                np.min(segment)
            ])

        # 3. 简单的循环特征
        # 计算20个等距点的值
        step = len(audio) // 20
        point_values = [audio[i * step] for i in range(20)]

        # 4. 简单的能量特征
        energy = np.sum(audio ** 2) / len(audio)

        # 组合所有特征
        features = np.array([mean, std, max_val, min_val, energy] + segment_features + point_values)

        return features

    except Exception as e:
        print(f"提取特征时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 加载数据并提取特征
def load_data(data_path):
    features = []
    labels = []

    if not os.path.exists(data_path):
        print(f"数据路径不存在: {data_path}")
        return None, None

    # 遍历所有情感文件夹
    for emotion in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        # 检查这个情感是否在我们的目标列表中
        if emotion not in Config.EMOTIONS:
            continue

        print(f"处理情感类别: {emotion}")
        # 遍历该情感类别下的所有音频文件
        audio_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        for audio_file in tqdm(audio_files):
            file_path = os.path.join(emotion_path, audio_file)

            # 提取特征
            feature = extract_features_simple(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)

    print(f"加载了 {len(features)} 个样本")
    return np.array(features), np.array(labels)


# 训练模型
def train_model(model_type='rf'):
    # 获取数据集路径
    data_path = input("输入数据集文件夹路径 (情感作为子文件夹): ")

    # 加载数据
    features, labels = load_data(data_path)
    if features is None or len(features) == 0:
        print("没有加载到数据。请检查数据路径和格式。")
        return None

    # 编码标签
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # 特征缩放
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, encoded_labels, test_size=0.2, random_state=42
    )

    # 选择模型
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        print("使用随机森林分类器...")
    else:
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        print("使用神经网络分类器...")

    # 训练模型
    print("开始训练模型...")
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"模型准确率: {accuracy:.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 可视化混淆矩阵
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)

    # 添加数值标签
    thresh = cm_normalized.max() / 2.0
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

    # 保存模型和预处理器
    with open(Config.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    with open(Config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    with open(Config.ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"模型已保存为 {Config.MODEL_PATH}")

    return model


# 预测单个音频文件的情感
def predict_emotion(audio_file):
    # 加载模型和预处理器
    if not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.ENCODER_PATH):
        print("模型或编码器未找到。请先训练模型。")
        return

    with open(Config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    with open(Config.ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    # 提取特征
    features = extract_features_simple(audio_file)

    if features is None:
        print(f"无法处理文件: {audio_file}")
        return

    # 特征缩放
    scaled_features = scaler.transform(features.reshape(1, -1))

    # 预测情感
    prediction = model.predict_proba(scaled_features)[0]
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

    return emotion, confidence


# 批量预测文件夹中的音频情感
def batch_predict_emotions(folder_path):
    # 加载模型和预处理器
    if not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.ENCODER_PATH):
        print("模型或编码器未找到。请先训练模型。")
        return

    with open(Config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    with open(Config.ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    results = []
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not audio_files:
        print("没有找到音频文件。")
        return

    print(f"分析文件夹中的 {len(audio_files)} 个音频文件...")

    for audio_file in tqdm(audio_files):
        file_path = os.path.join(folder_path, audio_file)

        # 提取特征
        features = extract_features_simple(file_path)

        if features is None:
            print(f"无法处理文件: {audio_file}")
            continue

        # 特征缩放
        scaled_features = scaler.transform(features.reshape(1, -1))

        # 预测情感
        prediction = model.predict_proba(scaled_features)[0]
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
        plt.title('音频文件中的情感分布')
        plt.xlabel('情感')
        plt.ylabel('数量')
        plt.tight_layout()
        plt.show()


# 主函数
def main():
    print("===== 语音情感识别系统 (scikit-learn版本) =====")
    print("这个版本使用scikit-learn，无需TensorFlow")

    while True:
        print("\n选择操作:")
        print("1. 训练新模型 (随机森林)")
        print("2. 训练新模型 (神经网络)")
        print("3. 从文件预测情感")
        print("4. 批量分析文件夹中的音频")
        print("5. 退出")

        choice = input("输入选择 (1-5): ")

        if choice == '1':
            train_model('rf')
        elif choice == '2':
            train_model('mlp')
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