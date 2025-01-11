import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
from PIL import Image


class App:
    def __init__(self, root):
        self.root = root
        self.model = tf.keras.models.load_model('D:/Desktop/Rock-paper-scissors/model/model.h5')

        # 映射标签到文本类别
        self.label_map = {0: "paper", 1: "rock", 2: "scissors"}

        # 创建GUI界面
        self.create_widgets()

    def create_widgets(self):
        # 创建文件选择按钮和预测按钮
        self.file_button = tk.Button(self.root, text="Select file", command=self.select_file)
        self.file_button.pack()

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack()

        # 创建一个用于显示预测结果的文本框
        self.result_text = tk.Text(self.root)
        self.result_text.pack()

    def select_file(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename()

        # 读取numpy数组文件
        self.file_data = np.load(file_path)

    def predict(self):
        # 获取待预测文件并进行预测
        prediction = self.model.predict(self.file_data)

        # 将预测结果转换为标签
        predicted_label = np.argmax(prediction, axis=1)[0]

        # 根据映射表将预测结果转换为文本类别
        predicted_text = self.label_map.get(predicted_label, "unknown")

        # 显示预测结果
        self.result_text.delete('1.0', 'end')
        self.result_text.insert('end', "Predicted label: {}\n".format(predicted_text))


# 创建主窗口并运行应用
root = tk.Tk()
app = App(root)
root.mainloop()