import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 定义函数，提取形状面积和周长特征
def extract_features(image):
    # 1. 计算形状、轮廓和面积特征
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 从彩色空间转换到灰度空间
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 将图像二值化以提取轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[0])

    # 2. 计算形状的周长特征
    perimeter = cv2.arcLength(contours[0], True)

    # 3. 计算形状的方向特征
    moments = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(moments)
    hu_moments = np.squeeze(np.asarray(hu_moments))

    # 将所有特征拼接到相同长度的向量上
    features = np.array([area, perimeter]+ hu_moments.tolist())

    # 返回特征向量
    return features

# 定义主程序，训练模型并测试准确率
if __name__ == '__main__':
    # 读取训练集数据
    train_data = []
    train_labels = []
    path = 'D:/Desktop/Binary classification/data/train'
    for filename in os.listdir(path):
        if 'great' in filename:
            label = 1  # 合格数据
        else:
            label = 0  # 不合格数据
        image = cv2.imread(os.path.join(path, filename))
        feature = extract_features(image)
        train_data.append(feature)
        train_labels.append(label)

    # 将所有图像特征拼接到相同长度的向量上
    max_len = max([len(x) for x in train_data])
    for i in range(len(train_data)):
        train_data[i] = np.pad(train_data[i], (0, max_len - len(train_data[i])), mode='constant')

    # 将训练数据和标签转换为NumPy数组，并将数据类型指定为浮点型
    train_data = np.array(train_data).astype(np.float32)
    train_labels = np.array(train_labels).astype(np.int32)

    # 训练SVM模型
    svm = cv2.ml.SVM_create()  # 创建SVM模型对象
    svm.setType(cv2.ml.SVM_C_SVC)  # 指定SVM模型的类型为C-Support Vector Classification（C-SVC）
    svm.setKernel(cv2.ml.SVM_RBF)  # 设置SVM模型的核函数为高斯径向基函数（RBF），这是应用最广泛的SVM核函数之一，广泛用于非线性数据的分类
    svm.train(cv2.UMat(train_data), cv2.ml.ROW_SAMPLE, cv2.UMat(train_labels))  # cv2.ml.ROW_SAMPLE表示使用行向量作为每个训练样本的表示方式

    # 保存SVM模型
    svm.save('svm_model.xml')

    # 加载SVM模型
    svm = cv2.ml.SVM_load('svm_model.xml')

    # 测试模型
    test_data = []
    test_labels = []
    path = 'D:/Desktop/Binary classification/data/val'
    for filename in os.listdir(path):
        if 'great' in filename:
            label = 1  # 合格数据
        else:
            label = 0  # 不合格数据
        image = cv2.imread(os.path.join(path, filename))
        feature = extract_features(image)
        test_data.append(feature)
        test_labels.append(label)

    # 将所有图像特征拼接到相同长度的向量上
    max_len = max([len(x) for x in test_data])
    for i in range(len(test_data)):
        test_data[i] = np.pad(test_data[i], (0, max_len - len(test_data[i])), mode='constant')

    # 将测试数据和标签转换为NumPy数组，并将数据类型指定为浮点型
    test_data = np.array(test_data).astype(np.float32)
    test_labels = np.array(test_labels).astype(np.int32)

    # 对测试数据进行调整，以保证其与训练数据形状相同
    test_data = np.resize(test_data, (len(test_labels), max_len)).astype(np.float32)

    # 打印训练集数据和测试集数据的形状
    print('Train data shape:', train_data.shape)
    print('Test data shape:', test_data.shape)

    # 对测试数据进行分类
    _, result = svm.predict(test_data)

    # 将cv2.UMat转换为NumPy数组
    result = np.asarray(result)

    # 统计分类结果
    correct = 0
    for i in range(len(result)):
        if result[i] == test_labels[i]:
            correct += 1

    # 对训练数据进行分类
    _, train_result = svm.predict(train_data)

    # 将cv2.UMat转换为NumPy数组
    train_result = np.asarray(train_result)

    # 统计分类结果
    train_correct = 0
    for i in range(len(train_result)):
        if train_result[i] == train_labels[i]:
            train_correct += 1

    # 输出准确率等评价指标
    accuracy = correct / len(test_labels)
    train_accuracy = train_correct / len(train_labels)
    print('Train accuracy:', train_accuracy)
    print('Test accuracy:', accuracy)

    # 计算混淆矩阵
    cm = confusion_matrix(test_labels, result)

    # 可视化混淆矩阵
    plt.matshow(cm)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()