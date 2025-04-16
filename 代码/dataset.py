import re
import os
from torchtext import data
import jieba
import logging
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import pandas as pd
from torch.utils.data import Dataset, DataLoader


jieba.setLogLevel(logging.INFO)
regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')  # 去除's、@、#、
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
             'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
             'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
             'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
             'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
             'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
             'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
             'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
             'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
             'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
             'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'] + ['!', ',', '.', '?', '-s', '-ly', '</s> ', 's']


class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df["text"].tolist()
        self.label = df["label"].tolist()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


def word_cut(text):
    '''
        切分文本
    :param text: 一个需要切分的文本
    :return: 切分后的文本
    '''
    # text = " ".join([i for i in text.split() if (i[:4] != 'http') and (i[:1] != '#') and (i[:1] != '@')])
    text = regex.sub(' ', text)
    return [word.lower() for word in jieba.cut(text) if word.strip()]  # and (word not in stopwords)


def bert_word_cut(text):
    text = regex.sub(' ', text)
    return " ".join([word.lower() for word in jieba.cut(text) if word.strip() and (word not in stopwords)])


def get_k_path(all_data, train_index, val_index, dir_path, index_k):
    '''
        获得所有切分好的数据，然后返回切分好的路径
    :param all_data: 所有的数据的DataFrame格式
    :param train_index: 划分数据的train_index
    :param val_index: 划分数据的val_index
    :param dir_path: 划分数据的直连路径
    :param index_k: 第几折
    :return: 第k折数据的路径
    '''
    train, val = all_data.iloc[train_index], all_data.iloc[val_index]
    k_path = dir_path + str(index_k) + '/'
    if not os.path.isdir(k_path):
        os.makedirs(k_path)
    train.to_csv(k_path + "/train.csv", sep=',', index=False, encoding='utf-8', header=True)
    val.to_csv(k_path + "/val.csv", sep=',', index=False, encoding='utf-8', header=True)
    return k_path


def divided_data(all_data, split_path, divided_type='train_test_split', k=1):
    '''
        划分数据
    :param k: 第几折数据
    :param all_data: 待划分的数据
    :param split_path: 划分数据的文件夹
    :param divided_type: 可选：[train_test_split, KFold, StratifiedKFold]
    :return: 第k折数据的路径
    '''
    all_data = pd.DataFrame(all_data)
    dir_path = split_path + str(divided_type) + '/'
    if divided_type == 'train_test_split':
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        train, val = train_test_split(all_data, test_size=0.2, shuffle=True, random_state=20)
        train.to_csv(dir_path + "/train.csv", sep=',', index=False, encoding='utf-8', header=True)
        val.to_csv(dir_path + "/val.csv",  sep=',', index=False, encoding='utf-8', header=True)
        return dir_path
    elif divided_type == 'KFold':
        kf = KFold(n_splits=5, shuffle=True, random_state=20)
        index_k = 1
        k_paths = []
        for train_index, val_index in kf.split(X=all_data):
            k_paths.append(get_k_path(all_data, train_index, val_index, dir_path, index_k))
            index_k += 1
        return k_paths[k-1]
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
        X = all_data
        Y = all_data['label']
        k_paths = []
        index_k = 1
        for train_index, val_index in skf.split(X, Y):
            k_paths.append(get_k_path(all_data, train_index, val_index, dir_path, index_k))
            index_k += 1
        return k_paths[k - 1]


def get_dataset(path, divided_flag, text_field, label_field, split_dir, divided_type, k):
    '''
        获得训练集和验证集
    :param path: 需要处理的原始数据的路径
    :param divided_flag: 是否对数据进行了划分
    :param text_field: torchtext的text_field
    :param label_field: torchtext的label_field
    :param split_dir: 划分数据的路径
    :param divided_type: 对数据该怎么划分
    :param k: 划分数据后选取第几折数据
    :return: train, dev
    '''
    # 读取数据
    all_data = pd.read_csv(path, encoding='utf-8', header=0)
    # 文本分词
    text_field.tokenize = word_cut  # 分词 IndexError: index out of range in self

    if not divided_flag:
        # 划分数据
        split_path = divided_data(all_data, split_dir, divided_type, k)
    elif divided_flag and k and divided_type:
        split_path = split_dir + str(divided_type) + '/' + str(k) + '/'
    else:
        split_path = split_dir + str(divided_type) + '/'

    train, dev = data.TabularDataset.splits(
        path=split_path, format='csv', skip_header=True,
        train='train.csv', validation='val.csv',
        fields=[
            ('id', None),  # id
            ('text', text_field),
            ('label', label_field),
        ]
    )  # split train 和 Validation 划分；
    return train, dev


def get_bert_dataset(text_field, label_field, path, divided_flag, split_dir, divided_type, k, batch_size):
    # 读取数据
    all_data = pd.read_csv(path, encoding='utf-8', header=0)
    if not divided_flag:
        # 划分数据
        split_path = divided_data(all_data, split_dir, divided_type, k)
    elif divided_flag and k and divided_type:
        split_path = split_dir + str(divided_type) + '/' + str(k) + '/'
    else:
        split_path = split_dir + str(divided_type) + '/'
    # return train, dev
    # train_path = split_path + 'train.csv'
    # val_path = split_path + 'val.csv'
    #
    # df_train = pd.read_csv(train_path, encoding='utf-8', header=0)
    # df_train["text"] = df_train["text"].apply(bert_word_cut)
    # df_train = df_train[["text", "label"]]
    #
    # df_test = pd.read_csv(val_path, encoding='utf-8', header=0)
    # df_test["text"] = df_test["text"].apply(bert_word_cut)
    # df_test = df_test[["text", "label"]]
    #
    # train_data = MyDataset(df_train)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #
    # test_data = MyDataset(df_test)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train_dataset, dev_dataset = data.TabularDataset.splits(
        path=split_path, format='csv', skip_header=True,
        train='train.csv', validation='val.csv',
        fields=[
            ('username', None),
            ('text', text_field),
            ('label', label_field),
        ]
    )
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter = data.Iterator.splits(
        (train_dataset, dev_dataset),
        batch_sizes=(batch_size, len(dev_dataset)),
        sort_key=lambda x: len(x.text),
        device=-1, repeat=False, shuffle=True)
    return train_iter, dev_iter
    # return train_iter, dev_iter








