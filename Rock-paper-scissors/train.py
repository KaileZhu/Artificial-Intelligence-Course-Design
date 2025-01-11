import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


# 加载数据集
train_data = np.load("D:/Desktop/Rock-paper-scissors/data/data_numpy/train_data.npy")
train_labels = np.load("D:/Desktop/Rock-paper-scissors/data/data_numpy/train_labels.npy")

# 转换标签为独热编码
train_labels = to_categorical(train_labels, num_classes=3)

# 定义模型
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # softmax 激活函数
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),  # 使用 CategoricalCrossentropy 损失函数
              metrics=['accuracy'])

# 训练模型并获取相应指标
batch_size = 64
steps_per_epoch = train_data.shape[0] // batch_size
history = model.fit(train_data, train_labels, epochs=15, batch_size=batch_size, steps_per_epoch=steps_per_epoch)

# 将模型保存为单独的文件
model.save('D:/Desktop/Rock-paper-scissors/model/model.h5')

# 画出训练过程的模型指标折线图
accuracy = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
plt.plot(epochs, loss, 'r-', label='Training loss')
plt.title('Training accuracy and loss')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()

# 输出模型结构图
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)