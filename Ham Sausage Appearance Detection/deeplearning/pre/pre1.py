import cv2
import glob

# 获取文件夹中所有bmp图片的文件名
image_files = glob.glob('D:\Desktop\Appearance Inspection of Ham Sausages\data\exam\*.bmp')

# 面积阈值
area_threshold = 40000

def intersect(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    if x1 > x2 + w2 or x2 > x1 + w1:
        return False
    if y1 > y2 + h2 or y2 > y1 + h1:
        return False
    return True

# 处理每个图片，并进行分类
for image_file in image_files:
    # 获取当前图片的文件名
    img = cv2.imread(image_file)

    # 获取亮度通道
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # 提取Y通道
    y_channel = ycrcb[:, :, 0]

    # 计算Y通道的均值和标准差
    y_mean, y_std = cv2.meanStdDev(y_channel)

    # 根据Y通道的均值选择二值化类型
    if y_mean > 128:
        # 浅色背景，使用cv2.THRESH_BINARY_INV
        _, thresh = cv2.threshold(y_channel, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        # 深色背景，使用cv2.THRESH_BINARY
        _, thresh = cv2.threshold(y_channel, 50, 255, cv2.THRESH_BINARY )

    # 生成结构元素用于腐蚀和膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # 对图像进行腐蚀和膨胀
    erosion = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # 找到轮廓并保存在deal文件中
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > area_threshold:
            # 检查框是否与之前的框相交
            intersecting = False
            for bbox in bboxes:
                if intersect(bbox, (x, y, w, h)):
                    intersecting = True
                    break
            if not intersecting:
                x_adjust = -20
                y_adjust = -20
                w_adjust = 40
                h_adjust = 50
                bboxes.append((x, y, w, h))
                cv2.rectangle(img, (x + x_adjust, y + y_adjust),
                              (x + x_adjust + w + w_adjust, y + y_adjust + h + h_adjust), (0, 255, 0), 2)
                # 提取火腿肠图像，并进行分类
                ham_img = img[y+y_adjust:y+y_adjust+h+h_adjust, x+x_adjust:x+x_adjust+w+w_adjust]
                ham_img = cv2.resize(ham_img, (224, 224))  # 调整成224*224
    # 显示结果
    cv2.imwrite('D:\Desktop\Appearance Inspection of Ham Sausages\data\deal\\' + str(image_file.split('\\')[-1]), img)
