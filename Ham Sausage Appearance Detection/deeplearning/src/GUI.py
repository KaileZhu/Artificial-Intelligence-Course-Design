import tkinter as tk
from tkinter import filedialog
import os
import torch
from PIL import Image
from torchvision import transforms
from model import Classifier, label_list

# 判断是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class App:
    def __init__(self, root):
        self.root = root

        # 创建GUI界面
        self.create_widgets()

    def create_widgets(self):
        # 创建文件选择按钮和预测按钮
        self.file_button = tk.Button(self.root, text="Select image", command=self.select_image)
        self.file_button.pack()

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack()

        # 创建一个用于显示预测结果的标签
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

        # 加载已经保存好的模型的权重和偏置
        self.classifier = Classifier(num_classes=3)
        self.classifier.load_state_dict(torch.load('D:/Desktop/deeplearning/model/model.pkl'))
        self.classifier.to(device)

        # 将模型设置为评估模式，关闭Dropout和BN层
        self.classifier.eval()

        # 数据预处理
        self.test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def select_image(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename()

        # 读取图像文件并进行预处理
        self.image = Image.open(file_path)
        self.image_tensor = self.test_transforms(self.image).unsqueeze(0).to(device)

    def predict(self):
        # 进行模型预测
        with torch.no_grad():
            outputs = self.classifier(self.image_tensor)
            _, prediction = torch.max(outputs, 1)

        # 显示预测结果
        predicted_label = label_list[prediction.item()]
        self.result_label.configure(text=predicted_label)

# 创建主窗口并运行应用
root = tk.Tk()
app = App(root)
root.mainloop()