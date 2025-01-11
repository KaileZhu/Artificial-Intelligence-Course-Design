import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import joblib  # 导入joblib库用于保存模型

image_dir = "D:/Desktop/gupao_degree/gupao/train"
image_files = os.listdir(image_dir)
images = [cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_GRAYSCALE) for f in image_files if f.endswith('.png')]


def extract_features(image):
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)

    if len(contours[0]) < 5:  # 如果轮廓中的像素点数小于5，直接返回0
        orientation = 0
    else:  # 否则，计算轮廓的椭圆拟合结果，并从中获取“形状的方向”这个特征
        ellipse = cv2.fitEllipse(contours[0])
        orientation = ellipse[2]

    return [area, perimeter, orientation]

features = [extract_features(image) for image in images]

# 使用k均值聚类算法对图像进行分组
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)
labels = kmeans.labels_

# 保存聚类模型
model_file = "kmeans_model_gu.joblib"
joblib.dump(kmeans, model_file)

# 显示聚类的结果并将其分类保存
cluster1_dir = "D:/Desktop/gupao_degree/gupao/train_cluster1"
cluster2_dir = "D:/Desktop/gupao_degree/gupao/train_cluster2"
cluster3_dir = "D:/Desktop/gupao_degree/gupao/train_cluster3"
if not os.path.exists(cluster1_dir):
    os.makedirs(cluster1_dir)
if not os.path.exists(cluster2_dir):
    os.makedirs(cluster2_dir)
if not os.path.exists(cluster3_dir):
    os.makedirs(cluster3_dir)

for i in range(num_clusters):
    cluster = np.where(labels == i)[0]
    print(f"Cluster {i+1}: {[image_files[j] for j in cluster]}")
    if i == 0:  # 将第一类图像保存到cluster1_dir
        for j in cluster:
            img_color = cv2.cvtColor(images[j], cv2.COLOR_GRAY2BGR)
            filename, ext = os.path.splitext(image_files[j])  # 使用os.path.splitext()获取文件名和扩展名
            cv2.imwrite(os.path.join(cluster1_dir, f"{filename}.png"), img_color)  # 添加文件后缀名".png"，保存
    elif i == 1:  # 将第二类图像保存到cluster2_dir
        for j in cluster:
            img_color = cv2.cvtColor(images[j], cv2.COLOR_GRAY2BGR)
            filename, ext = os.path.splitext(image_files[j])  # 使用os.path.splitext()获取文件名和扩展名
            cv2.imwrite(os.path.join(cluster2_dir, f"{filename}.png"), img_color)  # 添加文件后缀名".png"，保存
    else:  # 将第三类图像保存到cluster3_dir
        for j in cluster:
            img_color = cv2.cvtColor(images[j], cv2.COLOR_GRAY2BGR)
            filename, ext = os.path.splitext(image_files[j])  # 使用os.path.splitext()获取文件名和扩展名
            cv2.imwrite(os.path.join(cluster3_dir, f"{filename}.png"), img_color)  # 添加文件后缀名".png"，保存

# 显示空间特征与数量特征之间的关系
x = np.array([f[0] for f in features])
y = np.array([f[1] for f in features])
fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c=labels)
legend = ax.legend(*scatter.legend_elements(), loc="lower right", title="Clusters")
plt.show()