from PIL import Image
from PIL.ExifTags import TAGS
import os

def rotate_image(filepath):
    image = Image.open(filepath)
    exifdata = image._getexif()
    orientation = None
    if exifdata:
        for tag, value in exifdata.items():
            if TAGS.get(tag) == 'Orientation':
                orientation = value
                break

    if orientation:
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)

    return image

# 要调整大小的文件夹路径
folder_path = r"F:\python_code\Gesture recognition\data\unknow"

# 遍历文件夹内所有文件
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    if os.path.isfile(filepath):
        # 旋转图片
        image = rotate_image(filepath)
        # 调整图片大小
        image_resized = image.resize((300, 300))
        # 保存调整后的图片
        save_filepath = os.path.join(folder_path, filename)
        image_resized.save(save_filepath)