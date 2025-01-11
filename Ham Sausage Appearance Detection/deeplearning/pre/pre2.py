import os
import cv2
import glob
import numpy as np

# 获取文件夹中所有jpg图片的文件名
image_files = glob.glob('D:/Desktop/Appearance Inspection of Ham Sausages/data/deal/*.bmp')

# 面积阈值
area_threshold = 40000

# 遍历每张图片
for image_file in image_files:
    # 加载图片
    img = cv2.imread(image_file)

    # 找到所有绿色矩形框的轮廓
    contours, _ = cv2.findContours(cv2.inRange(img, (0, 255, 0), (0, 255, 0)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历每一个轮廓
    for i, contour in enumerate(contours):
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 如果轮廓面积大于阈值，则分割出子图像
        if area > area_threshold:
            # 获取轮廓的外接矩形框
            x, y, w, h = cv2.boundingRect(contour)

            # 创建一个空的掩膜来存储前景图像
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            # 填充掩膜
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # 将子图像保存到子文件夹中
            directory = "D:/Desktop/Appearance Inspection of Ham Sausages/data/Sub_images"
            if not os.path.exists(directory):
                os.makedirs(directory)

            # 将子图像保存到子文件夹中
            sub_image_path = directory + "\\" + str(image_file.split('\\')[-1][:-4]) + '_sub_img' + str(i+1) + '.png'
            cv2.imwrite(sub_image_path, cv2.bitwise_and(img, img, mask=mask)[y:y+h, x:x+w])