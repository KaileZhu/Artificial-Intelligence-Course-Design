import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

# 三种类型标签
label_list = ['合格', '弯曲缺陷', '鼓泡缺陷']

# 更改图片的大小，便于后续训练
width = 224
height = 224

# 定义对训练数据的预处理操作，包括图像大小缩放、随机裁剪、随机水平翻转、转换为张量和标准化处理
train_transform = transforms.Compose([
    transforms.Resize(256),  # 将图像按比例缩放到256像素大小
    transforms.RandomCrop(224),  # 从缩放后的图像中随机裁剪一个224像素大小的图像
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，增加数据的多样性
    transforms.ToTensor(),  # 将图像转换成张量形式，便于后续的操作
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 对图像进行标准化处理，使其在各个通道上的像素值均值为0，方差为1，增强了模型的训练和泛化能力
])

# 定义对测试数据的预处理操作，包括图像大小缩放、中心裁剪、转换为张量和标准化处理
test_transform = transforms.Compose([
    transforms.Resize(256),  # 将图像按比例缩放到256像素大小
    transforms.CenterCrop(224),  # 从图像中心裁剪一个224像素大小的图像
    transforms.ToTensor(),  # 将图像转换成张量形式，便于后续的操作
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 对图像进行标准化处理，使其在各个通道上的像素值均值为0，方差为1，增强了模型的训练和泛化能力
])

class MyDataset(data.Dataset):
    def __init__(self, dir_path, train=True):
        self.dir_path = dir_path
        self.train = train
        self.label_dict = {label_list[i]: i for i in range(len(label_list))}
        self.data_paths, self.labels = self.get_data_paths_labels()
        self.transform = train_transform if train else test_transform

    def get_data_paths_labels(self):
        data_paths = []
        labels = []
        images_dir = os.path.join(self.dir_path, 'train' if self.train else 'test')
        for label_dir in os.listdir(images_dir):
            label = self.label_dict[label_dir]
            label_path = os.path.join(images_dir, label_dir)
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                data_paths.append(file_path)
                labels.append(label)
        return data_paths, labels

    def __getitem__(self, index):
        file_path = self.data_paths[index]
        img = Image.open(file_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

# 判断是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将数据集移动到GPU上（num_workers默认为0表示不使用多线程）
train_dataset = MyDataset('D:/Desktop/deeplearning/data', train=True)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)