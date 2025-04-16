import pandas as pd
import numpy as np


class DataProcess:
    '''
        注意英文数据集有大小写转换和形式转换,在此认为不同词性的影响是相同的
        1. 这个数据集有用的列：['username', 'user id', 'content', 'sentiment'];
        2. 有的数据是没有'sentiment'的，得去掉;
    '''
    def __init__(self):
        self.__hcr_train_path = './data/HCR/train/orig/hcr-train.csv'
        self.__hcr_test_path = './data/HCR/test/orig/hcr-test.csv'
        self.__hcr_dev_path = './data/HCR/dev/orig/hcr-dev.csv'

        self.__processed_data_path = './data/HCR/processed_data.csv'

    @staticmethod
    def __read_csv(path):
        data = pd.read_csv(path, header=0, encoding='utf-8')
        return data

    @staticmethod
    def __store_csv_data(data, path):
        data = pd.DataFrame(data)
        data.to_csv(path, header=True, index=False, encoding='utf-8', mode='w')
        print('-' * 20 + 'File have been store.' + '-' * 20)

    @staticmethod
    def __del_blank(s):
        return s.strip()

    def data_cleansing(self):
        # 1. 取出数据
        train_data = self.__read_csv(self.__hcr_train_path)
        test_data = self.__read_csv(self.__hcr_test_path)
        dev_data = self.__read_csv(self.__hcr_dev_path)
        need_data_train = train_data.loc[:, ['username', 'user id', 'content', 'sentiment']]  # 1498
        need_data_test = test_data.loc[:, ['username', 'user id', 'content', 'sentiment']]  # 1500
        need_data_dev = dev_data.loc[:, ['username', 'user id', 'content', 'sentiment']]  # 1617

        # 2. nan值的处理
        need_data_train = need_data_train[pd.notnull(need_data_train['sentiment'])]  # 839
        need_data_test = need_data_test[pd.notnull(need_data_test['sentiment'])]  # 839
        need_data_dev = need_data_dev[pd.notnull(need_data_dev['sentiment'])]  # 837

        # print(need_data_dev.describe())  # 描述性分析
        # print(pd.isnull(need_data_train).any(1).nonzero()[0])  # 找出为nan的所有index  [  0  21 240] username
        # print(need_data_train.iloc[list(pd.isnull(need_data_train).any(1).nonzero()[0]), :])  # 找出为nan的行

        need_data_train = need_data_train.fillna('padding')
        # print(need_data_train.iloc[[0, 21, 240], :])  输出下看下填充的效果

        # 3. 数据整理与存储
        train_arr = need_data_train.values
        test_arr = need_data_test.values
        dev_arr = need_data_dev.values

        # print(train_arr.shape, test_arr.shape, dev_arr.shape)
        finally_data = np.vstack((train_arr, test_arr, dev_arr))
        # print(finally_data.shape)
        finally_data = pd.DataFrame(finally_data,
                                    index=list(np.arange(int(finally_data.shape[0]))),
                                    columns=['username', 'user id', 'content', 'sentiment'])
        finally_data['sentiment'] = finally_data['sentiment'].apply(self.__del_blank)
        p_n_data = finally_data[(finally_data['sentiment'] == 'positive') | (finally_data['sentiment'] == 'negative')]  # positive  negative
        label_dict = {
            'negative': 0,
            'positive': 1
        }
        p_n_data['sentiment'] = p_n_data['sentiment'].map(label_dict)
        self.__store_csv_data(p_n_data, self.__processed_data_path)


dataprocess = DataProcess()
dataprocess.data_cleansing()

