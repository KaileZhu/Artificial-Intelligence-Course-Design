import os
import shutil

folder_path = "D:/Desktop/Binary classification/data/test/Qualified"  # 需要重命名的目录路径
prefix = "Qualified"  # 新文件名的前缀

files = os.listdir(folder_path)  # 获取目录中的所有文件名

for i, filename in enumerate(files):
    _, ext = os.path.splitext(filename)  # 分离文件名和扩展名
    new_filename = f"{prefix}{i+1:03d}{ext}"  # 生成新文件名
    old_path = os.path.join(folder_path, filename)  # 原文件的完整路径
    new_path = os.path.join(folder_path, new_filename)  # 新文件的完整路径
    shutil.move(old_path, new_path)  # 重命名文件
