import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow.keras as keras

# 加载模型
model = keras.models.load_model("F:\python_code\Gesture recognition\model\model_best.h5")

img_dir = "F:/python_code/Gesture recognition/data/unknown/"
img_list = os.listdir(img_dir)

for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    # 将概率值最大的下标作为模型的预测结果
    result = np.argmax(result, axis=1)
    if result == 0:
        result = 'paper'
        print(f"{img_name}的分类结果为: ", result)
    elif result == 1:
        result = 'rock'
        print(f"{img_name}的分类结果为: ", result)
    elif result == 2:
        result = 'scissors'
        print(f"{img_name}的分类结果为: ", result)
