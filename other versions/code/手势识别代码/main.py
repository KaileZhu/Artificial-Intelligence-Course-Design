from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 设计CNN的模型框架
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

# 编译模型，设置损失函数、优化器和评估指标等参数
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# 加载数据，并进行数据增强
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = 'F:/python_code/Gesture recognition/data/train'
val_dir = 'F:/python_code/Gesture recognition/data/val'
test_dir = 'F:/python_code/Gesture recognition/data/test'

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(300, 300),
                                                    batch_size=32,
                                                    class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(300, 300),
                                                batch_size=32,
                                                class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(300, 300),
                                                  batch_size=32,
                                                  class_mode='categorical')
# 模型保存，定义回调函数
filepath = r"F:\python_code\Gesture recognition\model\model_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# 开始训练模型
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=val_generator, callbacks=[checkpoint])
# 获取训练集和验证集上记录下来的准确率和损失函数值
train_acc = model.history.history['accuracy']
valid_acc = model.history.history['val_accuracy']
train_loss = model.history.history['loss']
valid_loss = model.history.history['val_loss']

# 绘制训练集和验证集上的准确率曲线
plt.plot(train_acc, label='train_acc')
plt.plot(valid_acc, label='valid_acc')
plt.legend()
plt.show()

# 绘制训练集和验证集上的损失函数曲线
plt.plot(train_loss, label='train_loss')
plt.plot(valid_loss, label='valid_loss')
plt.legend()
plt.show()

# 对模型进行评估
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
print('Test precision:', test_precision)
print('Test recall:', test_recall)

plot_model(model=model, to_file='gesture.png', show_shapes=True)


