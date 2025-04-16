import pandas as pd
import numpy as np


class DataProcess(object):
    def __init__(self):
        self.path = './data/OMD/debate08_sentiment_tweets.tsv'
        self.processed_path = './data/OMD/processed_data.csv'

    @staticmethod
    def __read_tsv(path):
        data = pd.read_csv(path, sep='	', header=0, encoding='utf-8')
        return data

    @staticmethod
    def __struct_labels(labels_array):
        '''
            返回投票后的列表
        :param labels_array:
        :return:
        '''
        labels_list = []
        for i in range(int(labels_array.shape[0])):
            temp_data = []
            for j in range(int(labels_array.shape[1])):
                if np.isnan(labels_array[i][j]):
                    pass
                temp_data.append(labels_array[i][j])
            arr_num = max(temp_data, key=temp_data.count)
            labels_list.append(arr_num - 1)
        return labels_list

    @staticmethod
    def __store_csv_data(data, path):
        data = pd.DataFrame(data)
        data.to_csv(path, header=True, index=False, encoding='utf-8', mode='w')
        print('-' * 20 + 'File have been store.' + '-' * 20)

    def fast_process(self):
        data = self.__read_tsv(self.path)
        print(len(data))
        labels_array = data[['rating.1',
                             'rating.2',
                             'rating.3',
                             'rating.4',
                             'rating.5',
                             'rating.6',
                             'rating.7',
                             'rating.8']].values
        assert labels_array.shape == (3238, 8)
        labels_list = self.__struct_labels(labels_array)  # 获得labels

        # 去除不用的列
        del data['pub.date.GMT']
        # del data['author.name']
        del data['author.nickname']
        del data['rating.1']
        del data['rating.2']
        del data['rating.3']
        del data['rating.4']
        del data['rating.5']
        del data['rating.6']
        del data['rating.7']
        del data['rating.8']

        # 追加labels_list
        data.insert(2, 'labels', pd.Series(np.array(labels_list), dtype=int))
        data = data[(data['labels'] == 1) | (data['labels'] == 0)]  # 0 1
        self.__store_csv_data(data, self.processed_path)


data_process = DataProcess()
data_process.fast_process()



