import os
import shutil

# 定义第一个和第二个文件夹的路径
# folder1_path = "D:/Desktop/Rock-paper-scissors/data/new_data/train/paper"
# folder2_path = "D:/Desktop/Rock-paper-scissors/data/new_data/val/paper"
# folder1_path = "D:/Desktop/Rock-paper-scissors/data/new_data/train/rock"
# folder2_path = "D:/Desktop/Rock-paper-scissors/data/new_data/val/rock"
folder1_path = "D:/Desktop/Rock-paper-scissors/data/new_data/train/scissors"
folder2_path = "D:/Desktop/Rock-paper-scissors/data/new_data/val/scissors"

# 创建存放文件夹的位置
os.makedirs(folder1_path, exist_ok=True)
os.makedirs(folder2_path, exist_ok=True)

# 定义目标文件夹路径
# source_folder_path = "D:/Desktop/Rock-paper-scissors/data/data/paper"
# source_folder_path = "D:/Desktop/Rock-paper-scissors/data/data/rock"
source_folder_path = "D:/Desktop/Rock-paper-scissors/data/data/scissors"

# 定义计数变量
count = 0

# 遍历目标文件夹中的所有文件
for filename in os.listdir(source_folder_path):
    # 判断文件是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 获取文件的绝对路径
        filepath = os.path.join(source_folder_path, filename)
        # 按照需求将文件复制到第一个或第二个文件夹中
        count += 1
        if count % 10 in [8, 9]:  # 第9、10张图放入第二个文件夹
            shutil.copy(filepath, folder2_path)
        else:  # 其余图片放入第一个文件夹
            shutil.copy(filepath, folder1_path)