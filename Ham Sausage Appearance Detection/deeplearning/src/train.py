import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from dataset import MyDataset, label_list
from model import Classifier

# 超参数
epochs = 150
batch_size = 32
lr = 0.013
num_classes = len(label_list)

# 判断是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



label_list = ['合格', '弯曲缺陷', '鼓泡缺陷']

def train():
    # 加载数据集
    train_dataset = MyDataset('D:/Desktop/deeplearning/data', train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 加载测试集
    test_dataset = MyDataset('D:/Desktop/deeplearning/data', train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 定义分类器模型
    classifier = Classifier(num_classes=num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # 记录训练过程中的训练损失、训练集准确率和测试集准确率
    train_losses = []
    train_accs = []
    test_accs = []

    # 训练模型
    for epoch in range(epochs):
        classifier.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 统计训练集准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 计算本轮训练结果
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # 统计测试集准确率
        classifier.eval()
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        # 计算本轮测试结果
        test_accuracy = test_correct / test_total
        test_accs.append(test_accuracy)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2%}, Test Acc: {:.2%}'.format(epoch+1, epochs, train_loss, train_accuracy, test_accuracy))

    # 保存模型
    torch.save(classifier.state_dict(), 'D:/Desktop/deeplearning/model/model.pkl')

    # 绘制折线图
    plt.plot(train_losses, label='train loss')
    plt.plot(train_accs, label='train accuracy')
    plt.plot(test_accs, label='test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Train/Test Loss and Accuracy')
    plt.legend()
    plt.show()

    # 计算最终的混淆矩阵
    classifier.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels += labels.cpu().tolist()
            all_preds += predicted.cpu().tolist()

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 显示混淆矩阵
    fig, axs = plt.subplots(figsize=(7, 7))
    axs.imshow(conf_matrix)
    for i in range(len(label_list)):
        for j in range(len(label_list)):
            axs.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')
    plt.xticks(range(len(label_list)), label_list, rotation=45)
    plt.yticks(range(len(label_list)), label_list)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    plt.show()


if __name__ == '__main__':
    train()