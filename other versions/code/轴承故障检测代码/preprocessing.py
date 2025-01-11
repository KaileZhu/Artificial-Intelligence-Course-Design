from scipy.io import loadmat
import scipy.io
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同


def prepro(d_path, length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28, ):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enc=True,
                                                                    enc_step=28)
    ```
    """

    # 获得该文件夹下所有.mat文件名
    def read_mat_folder(folder_path):
        """
        读取一个包含多个Matlab文件的文件夹，获取其中所有文件中的data矩阵，并将这些矩阵存储在一个列表中返回。

        Args:
            folder_path: 包含多个Matlab文件的文件夹路径。

        Returns:
            data_matrices: 一个列表，其中包含了所有Matlab文件中的data矩阵。
        """
        # 获取文件夹中所有Matlab文件
        mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

        # 遍历每个Matlab文件并获取其data矩阵
        data_matrices = []
        for file_name in mat_files:
            mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
            data_matrix = mat['data']
            data_matrices.append(data_matrix)

        return data_matrices


    def split_data(data_list, slice_rate=rate[1] + rate[2]):
        """
        将数据列表分为切分好的训练数据和测试数据
        :param data_list: 数据列表，包含9个元素，每个元素的构成为（时序×特征维度为3）
        :param slice_rate: 验证集以及测试集所占的比例
        :param enc: 是否进行编码
        :param length: 切分出的样本长度
        :param enc_step: 编码步长，用于控制采样间隔
        :param number: 训练样本数量
        :return: 切分好的训练数据和测试数据
        """
        # 训练、测试集占比
        train_rate = 1 - slice_rate

        Train_Samples = []
        Test_Samples = []

        for slice_data in data_list:
            # 数据长度
            all_lenght = len(slice_data)
            # 计算切分位置
            end_index = int(all_lenght * train_rate)
            samp_train = int(number * train_rate)
            Train_sample = []
            Test_sample = []

            if enc:
                # 编码训练数据
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                # 随机切分训练数据
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 随机切分测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_sample.append(sample)

            Train_Samples.append(Train_sample)
            Test_Samples.append(Test_sample)

        return Train_Samples, Test_Samples

    def add_labels(data):
        """
        这是一个用于添加标签的函数，输入参数为一个数据集data，其中每个slice_data表示一个类别内的数据
        函数首先初始化X和Y，X用于保存所有数据，Y用于保存所有数据对应的标签
        label用于表示当前处理的类别，初始为0
        遍历每个slice_data，将其中所有数据添加到X中，并在Y中添加相对应的标签
        标签的值为label，每次处理完一个slice_data后label值加1
        返回X和Y
        """
        X = []
        Y = []
        label = 0
        for slice_data in data:
            X += slice_data
            lenx = len(slice_data)
            Y += [label] * lenx
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        """
        这是一个用于将标签进行one-hot编码的函数，输入参数为Train_Y和Test_Y，分别表示训练集和测试集中的标签
        首先将Train_Y和Test_Y转换为numpy array，并将二维数组的第二个维度统一设置为1，以方便后续处理
        接着使用sklearn中的preprocessing库中的OneHotEncoder函数进行one-hot编码
        对于训练集中的标签Train_Y，首先使用Encoder.fit函数对其进行拟合，以获取该标签中不同的取值，即不同的类别
        然后使用Encoder.transform函数对Train_Y进行one-hot编码，并将结果转换为numpy array类型
        对于测试集中的标签Test_Y，同样使用之前拟合得到的Encoder对其进行one-hot编码，并将结果转换为numpy array类型
        最后将编码后的Train_Y和Test_Y转换为int32类型的数组，并将其作为输出结果返回
        """
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def standardize_data(train_data, test_data):
        """
        对训练集和测试集进行标准化
        :param train_data: 训练集数据，列表类型，每个元素是一个形状为 (time_steps, features) 的列表
        :param test_data: 测试集数据，列表类型，每个元素是一个形状为 (time_steps, features) 的列表
        :return: 标准化后的训练集和测试集
        """
        # 转换为 Numpy 数组，并将所有数据合并为一个大数组
        all_data = np.vstack(train_data + test_data)

        # 计算均值和标准差
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)

        # 训练集标准化
        train_data_norm = [(np.array(x) - mean) / std for x in train_data]

        # 测试集标准化
        test_data_norm = [(np.array(x) - mean) / std for x in test_data]

        return train_data_norm, test_data_norm

    def valid_test_slice(Test_X, Test_Y):
        """
        这是一个用于将测试集划分为验证集和测试集的函数，输入参数为Test_X和Test_Y，分别表示测试集中的数据和标签
        首先将Test_X和Test_Y转换为numpy array类型
        接着通过划分比例rate[2]/(rate[1]+rate[2])计算出测试集中需要划分到验证集中的数据所占比例test_size
        使用sklearn中的StratifiedShuffleSplit函数，设置划分1次和测试集中数据和标签为X和Y，对其进行划分
        train_index和test_index分别表示划分后的训练集和测试集在原测试集中的索引位置
        对于划分后的数据，将训练集train_index对应的Test_X中的数据作为验证集X_valid，将对应的Test_Y中的标签作为验证集Y_valid
        将测试集test_index对应的Test_X中的数据作为测试集X_test，将对应的Test_Y中的标签作为测试集Y_test
        最后将划分好的X_valid, Y_valid, X_test, Y_test作为输出结果返回
        """
        Test_X = np.array(Test_X)
        Test_Y = np.array(Test_Y)
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = read_mat_folder(d_path)
    # 使用NumPy的array函数将data转换为NumPy数组
    data = np.array(data)
    # 将数据切分为训练集、测试集
    train, test = split_data(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = standardize_data(Train_X, Test_X)
    # 需要做一个数据转换，转换成np格式.
    Train_X = np.asarray(Train_X)
    Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y, data


if __name__ == "__main__":
    path = r'F:\python_code\Bearing failure diagnosis\data\train'
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y, data = prepro(d_path=path,
                                                                      length=1024,
                                                                      number=1000,
                                                                      normal=False,
                                                                      rate=[0.5, 0.25, 0.25],
                                                                      enc=True,
                                                                      enc_step=28)
