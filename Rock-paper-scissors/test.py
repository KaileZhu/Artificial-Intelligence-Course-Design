import tensorflow as tf
import numpy as np

def evaluate_model(model, data, labels):
    batch_size = 32
    num_samples = data.shape[0]
    predictions = np.zeros((num_samples, 3))
    for i in range(0, num_samples, batch_size):
        x_batch = data[i:i+batch_size]
        predictions[i:i+batch_size] = model.predict_on_batch(x_batch)

    # 映射标签到文本类别
    label_map = {0: "paper", 1: "rock", 2: "scissors"}

    # 将预测结果转换为标签并映射到文本类别
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_text = np.array([label_map.get(label, "unknown") for label in predicted_labels])

    # 输出每个文件的预测结果
    for i in range(len(predicted_text)):
        print("File:", i+1, "Predicted label:", predicted_text[i], "True label:", label_map.get(labels[i], "unknown"))

    # 计算并输出每个类别的准确率
    accuracy = {}
    for i in range(3):
        idx = np.where(labels == i)[0]
        acc = np.mean(predicted_labels[idx] == labels[idx])
        accuracy["class " + label_map[i]] = acc
    print("\nPer-class accuracy:", accuracy)

    # 计算并输出总体准确率和分类报告
    overall_accuracy = np.mean(predicted_labels == labels)
    print("\nOverall accuracy:", overall_accuracy)

if __name__ == '__main__':
    # 加载训练好的模型
    model = tf.keras.models.load_model('D:/Desktop/Rock-paper-scissors/model/model.h5')

    # 加载测试集数据
    exam_data = np.load("D:/Desktop/Rock-paper-scissors/data/data_numpy/exam_data.npy")
    exam_labels = np.load("D:/Desktop/Rock-paper-scissors/data/data_numpy/exam_labels.npy")

    evaluate_model(model, exam_data, exam_labels)