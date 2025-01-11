import os
import numpy as np
from PIL import Image

# 定义一些常
TRAIN_DIR = "D:/Desktop/Rock-paper-scissors/data/new_data/train"
# VAL_DIR = "D:/Desktop/Rock-paper-scissors/data/new_data/val"
# EXAM_DIR = "D:/Desktop/Rock-paper-scissors/data/exam"
PRE_DIR = "D:/Desktop/Rock-paper-scissors/data/dealPre"
IMG_SIZE = (224, 224)

# 定义标签映射
class_names = sorted(os.listdir(TRAIN_DIR))
label_map = {name: i for i, name in enumerate(class_names)}


# 定义函数：加载、处理图像并返回图像数据和标签
def load_and_process_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMG_SIZE)
    img = img.convert("RGB")  # 将图像转换为RGB模式
    img = np.array(img) / 255.0
    img = img.astype('float32')
    return img


# 定义函数：处理整个数据集
def preprocess(data_dir):
    data = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            img = load_and_process_image(image_path)

            # 将图像数据和标签保存到数组中
            data.append(img)
            labels.append(label_map[class_name])

    # 将数据和标签转换为numpy数组形式
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# 对训练集和验证集进行预处理
# train_data, train_labels = preprocess(TRAIN_DIR)
# val_data, val_labels = preprocess(VAL_DIR)
# exam_data, exam_labels = preprocess(EXAM_DIR)
pre_data, pre_labels = preprocess(TRAIN_DIR)

# 将处理后的图像数据保存为224x224大小的数组，同时将标签保存为Numpy数组形式的文件
# np.save('D:/Desktop/Rock-paper-scissors/data/data_numpy/train_data.npy', train_data)
# np.save('D:/Desktop/Rock-paper-scissors/data/data_numpy/train_labels.npy', train_labels)
# np.save('D:/Desktop/Rock-paper-scissors/data/data_numpy/val_data.npy', val_data)
# np.save('D:/Desktop/Rock-paper-scissors/data/data_numpy/val_labels.npy', val_labels)
# np.save('D:/Desktop/Rock-paper-scissors/data/data_numpy/exam_data.npy', exam_data)
# np.save('D:/Desktop/Rock-paper-scissors/data/data_numpy/exam_labels.npy', exam_labels)
np.save('D:/Desktop/Rock-paper-scissors/data/data_numpy/pre_data.npy', pre_data)
np.save('D:/Desktop/Rock-paper-scissors/data/data_numpy/pre_labels.npy', pre_labels)