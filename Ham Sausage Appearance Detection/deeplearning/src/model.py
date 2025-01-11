import torch
import torch.nn as nn
import torchvision.models as models
from torchviz import make_dot
import matplotlib.pyplot as plt

label_list = ['合格', '弯曲缺陷', '鼓泡缺陷']

# 判断是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = models.resnet18(pretrained=True)  # 采用预训练的resnet18模型
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.to(self.device)

    def forward(self, x):
        """前向传播函数"""
        out = self.resnet(x.to(self.device, dtype=self.resnet.fc.weight.dtype))
        return out

# clf = Classifier(num_classes=len(label_list)).to(device)
# x = torch.zeros((1, 3, 224, 224)).to(device)
# out = clf(x)
# make_dot(out, params=dict(clf.named_parameters())).render("model", format="png")
# plt.imshow(plt.imread("model.png"))