import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MyDataset, label_list
from model import Classifier

# 定义超参数
batch_size = 32
num_classes = len(label_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
    # 加载测试集
    test_dataset = MyDataset('D:/Desktop/deeplearning/data', train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 加载模型
    classifier = Classifier(num_classes=num_classes).to(device)
    classifier.load_state_dict(torch.load('D:/Desktop/deeplearning/model/model.pkl'))

    # 测试模型
    classifier.eval()
    pred_dict = {label: [] for label in label_list}
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 记录每个类别的预测结果
            for i in range(len(labels)):
                label = label_list[labels[i]]
                label_pred = label_list[predicted[i]]
                pred_dict[label].append((label_pred == label))
                # 显示当前样本的输出情况
                print(
                    "Sample No.{} - True Label: {} - Predicted Label: {}".format((batch_size * total) + (i + 1), label,
                                                                                 label_pred))  # 加上(batch_size*total)是为了方便维护编号

    import matplotlib.pyplot as plt

    # 输出每个类别的准确率并记录
    class_accuracies = []
    for label in ['合格', '弯曲缺陷', '鼓泡缺陷']:
        accuracy = sum(pred_dict[label]) / len(pred_dict[label])
        class_accuracies.append(accuracy)
        print("Accuracy for class {}: {:.2%}".format(label, accuracy))

    # 输出总准确率并记录
    accuracy = correct / total
    print("Overall accuracy: {:.2%}".format(accuracy))
    class_accuracies.append(accuracy)

    # 绘制条形图
    labels = ['Qualified', 'Bent Defects', 'Bubble Defects', 'Overall']
    plt.bar(labels, class_accuracies)
    plt.ylim(0, 1)  # 设置y轴范围
    plt.title('Accuracy by class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    # 在每个条形图上方加入其在纵坐标得数值
    for i, v in enumerate(class_accuracies):
        plt.text(i, v + 0.05, format(v, '.2%'), ha='center')
    plt.show()


if __name__ == '__main__':
    test()