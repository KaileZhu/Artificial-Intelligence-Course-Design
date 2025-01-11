import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Classifier

# 三种类型标签
label_list = ['合格', '弯曲缺陷', '鼓泡缺陷']

# 定义设备类型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载已经保存好的模型的权重和偏置
classifier = Classifier(num_classes=3).to(device)
classifier.load_state_dict(torch.load('D:/Desktop/deeplearning/model/model.pkl'))

# 将模型设置为评估模式，关闭Dropout和BN层
classifier.eval()

# 获取未标记的图像文件名列表
image_folder_path = 'D:/Desktop/deeplearning/data/examation'
image_filenames = os.listdir(image_folder_path)

# 数据预处理
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# 定义批大小和数据加载器
batch_size = 32
data_loader = DataLoader(image_filenames, batch_size=batch_size)

# 用于记录“合格”图像的数量
num_qualified = 0

# 分批处理数据，并计算分类准确率
for batch in data_loader:
    # 获取图像文件名列表
    batch_filenames = batch

    # 读取图像
    images = []
    for filename in batch_filenames:
        filepath = os.path.join(image_folder_path, filename)
        image = Image.open(filepath)
        images.append(image)

    # 数据预处理
    batch_images = [test_transforms(image) for image in images]
    batch_images = torch.stack(batch_images).to(device)

    # 进行模型预测
    with torch.no_grad():
        outputs = classifier(batch_images)
        _, predictions = torch.max(outputs, 1)


    # 修改图片名称，加上预测结果
    for i in range(len(batch_filenames)):
        filename = batch_filenames[i]
        extension = os.path.splitext(filename)[-1]
        label = label_list[predictions[i]]
        new_filename = f"{label}_{filename}"
        new_filepath = os.path.join(image_folder_path, new_filename)
        old_filepath = os.path.join(image_folder_path, filename)
        os.rename(old_filepath, new_filepath)
