import numpy as np
import scipy.io
import os
import tensorflow.keras as keras
import pandas as pd


# 加载模型
model = keras.models.load_model('F:\python_code\Bearing failure diagnosis\model\model_best.h5')

folder_path = 'F:\python_code\Bearing failure diagnosis\data\Test'
file_name = 'datatest'
mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
test = mat['TestDataArray']
test = np.transpose(test, (2, 0, 1))  # 将第一维转换为第二维，第二维转换为第三维，第三维转换为第一维
Y_test = model.predict(test)
# 将概率值最大的下标作为模型的预测结果
Y_test = np.argmax(Y_test, axis=1)

# 首先定义替换的字典，将值转换为字符串类型
replace_dict = {0: 'IR07', 1: 'BL07', 2: 'OR07', 3: 'IR14', 4: 'BL14', 5: 'OR14', 6: 'IR21', 7: 'BL21', 8: 'OR21'}
# 使用numpy中的vectorize函数对字典中的值进行类型转换
replace_func = np.vectorize(lambda x: str(x))
# 将Y_test转换为字符串类型
Y_test_str = replace_func(Y_test)
Y_test_str = Y_test_str.reshape((-1, 1))
# 使用numpy中的where函数进行替换操作
for key, value in replace_dict.items():
    Y_test_str = np.where(Y_test_str == str(key), value, Y_test_str)
# 创建Pandas数据帧
df = pd.DataFrame({'Prediction': Y_test_str.flatten()})
# 将数据帧写入CSV文件
df.to_csv('F:\python_code\Bearing failure diagnosis\predictions3.csv', index=False)
print("分类完成")
