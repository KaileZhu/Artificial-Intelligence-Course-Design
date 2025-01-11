import os
import random
import shutil

# 移动
in0 = r'F:\python_code\Gesture recognition\data\数据\paper'
out1 = r'F:\python_code\Gesture recognition\data\train\paper'
out2 = r'F:\python_code\Gesture recognition\data\val\paper'
out3 = r'F:\python_code\Gesture recognition\data\test\paper'

# 获取所有图片的文件名
filenames = os.listdir(in0)
# 随机打乱文件名顺序
random.shuffle(filenames)

# 计算训练集、验证集、测试集对应的图片数量
train_num = int(len(filenames) * 0.8)
val_num = int(len(filenames) * 0.1)
test_num = len(filenames) - train_num - val_num

# 确定训练集、验证集、测试集分别对应的文件名列表
train_files = filenames[:train_num]
val_files = filenames[train_num:train_num+val_num]
test_files = filenames[-test_num:]

# 创建输出目录1、2和3
os.makedirs(out1, exist_ok=True)
os.makedirs(out2, exist_ok=True)
os.makedirs(out3, exist_ok=True)

# 将文件名分配到三个目录
for dirname, file_list in zip([out1, out2, out3], [train_files, val_files, test_files]):
    for filename in file_list:
        shutil.copy(os.path.join(in0, filename), os.path.join(dirname, filename))

print(f"{train_num}张图片已放在{out1}中，{val_num}张图片已放在{out2}中，{test_num}张图片已放在{out3}中。")



# 移动
in0 = r'F:\python_code\Gesture recognition\data\数据\rock'
out1 = r'F:\python_code\Gesture recognition\data\train\rock'
out2 = r'F:\python_code\Gesture recognition\data\val\rock'
out3 = r'F:\python_code\Gesture recognition\data\test\rock'

# 获取所有图片的文件名
filenames = os.listdir(in0)
# 随机打乱文件名顺序
random.shuffle(filenames)

# 计算训练集、验证集、测试集对应的图片数量
train_num = int(len(filenames) * 0.8)
val_num = int(len(filenames) * 0.1)
test_num = len(filenames) - train_num - val_num

# 确定训练集、验证集、测试集分别对应的文件名列表
train_files = filenames[:train_num]
val_files = filenames[train_num:train_num+val_num]
test_files = filenames[-test_num:]

# 创建输出目录1、2和3
os.makedirs(out1, exist_ok=True)
os.makedirs(out2, exist_ok=True)
os.makedirs(out3, exist_ok=True)

# 将文件名分配到三个目录
for dirname, file_list in zip([out1, out2, out3], [train_files, val_files, test_files]):
    for filename in file_list:
        shutil.copy(os.path.join(in0, filename), os.path.join(dirname, filename))

print(f"{train_num}张图片已放在{out1}中，{val_num}张图片已放在{out2}中，{test_num}张图片已放在{out3}中。")

# 移动
in0 = r'F:\python_code\Gesture recognition\data\数据\scissors'
out1 = r'F:\python_code\Gesture recognition\data\train\scissors'
out2 = r'F:\python_code\Gesture recognition\data\val\scissors'
out3 = r'F:\python_code\Gesture recognition\data\test\scissors'

# 获取所有图片的文件名
filenames = os.listdir(in0)
# 随机打乱文件名顺序
random.shuffle(filenames)

# 计算训练集、验证集、测试集对应的图片数量
train_num = int(len(filenames) * 0.8)
val_num = int(len(filenames) * 0.1)
test_num = len(filenames) - train_num - val_num

# 确定训练集、验证集、测试集分别对应的文件名列表
train_files = filenames[:train_num]
val_files = filenames[train_num:train_num+val_num]
test_files = filenames[-test_num:]

# 创建输出目录1、2和3
os.makedirs(out1, exist_ok=True)
os.makedirs(out2, exist_ok=True)
os.makedirs(out3, exist_ok=True)

# 将文件名分配到三个目录
for dirname, file_list in zip([out1, out2, out3], [train_files, val_files, test_files]):
    for filename in file_list:
        shutil.copy(os.path.join(in0, filename), os.path.join(dirname, filename))

print(f"{train_num}张图片已放在{out1}中，{val_num}张图片已放在{out2}中，{test_num}张图片已放在{out3}中。")