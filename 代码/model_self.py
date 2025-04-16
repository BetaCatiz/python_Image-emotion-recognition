import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import sys
import re
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sm
import json
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


torch.manual_seed(0)  # 设置随机种子


# 数据集
class MyDataset(Dataset):
    def __init__(self, df, df_emb):
        self.data = df["text"].tolist()
        self.data_emb = df_emb["text"].tolist()
        self.label = df["label"].tolist()

    def __getitem__(self, index):
        data = self.data[index]
        data_emb = self.data_emb[index]
        label = self.label[index]
        return data, data_emb, label

    def __len__(self):
        return len(self.label)


class TrainBertCNNEmb(object):
    """
        训练BERT模型:
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    def __init__(self, data_set_name, k):
        self.regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')  # 去除's、@、#、
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
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
        self.train_path = './data/' + str(data_set_name) + '/split_data/StratifiedKFold/' + str(k) + '/train.csv'
        self.val_path = './data/' + str(data_set_name) + '/split_data/StratifiedKFold/' + str(k) + '/val.csv'
        self.emb_train_path = './data/' + str(data_set_name) + '/relation_embedding_data/StratifiedKFold/' + str(k) + '/train.csv'
        self.emb_val_path = './data/' + str(data_set_name) + '/relation_embedding_data/StratifiedKFold/' + str(k) + '/val.csv'

    def __english_f(self, text):
        text = self.regex.sub(' ', text)
        # text = " ".join([word.lower() for word in text.split() if (word.strip()) and (word.lower() not in self.stopwords)])
        temp = []
        for word in text.split():
            if (word.strip()) and (word.lower() not in self.stopwords) and (word[0] != '#') and (word[0] != '@'):
                if word[0] == '"':
                    temp.append(word[1:])
                elif word[-1] == '"' or word[-1:] == "!" or word[-1:] == "?":
                    temp.append(word[:-1])
                elif word[0:2] == "''":
                    temp.append(word[2:])
                elif word[-2:] == "''":
                    temp.append(word[:-2])
                else:
                    temp.append(word)
        text = " ".join(temp)
        return text

    @staticmethod
    def __embedding_f(text):
        text = eval(text)
        text = torch.FloatTensor(text)
        return text

    def __generate_data_loader(self, data, data_emb, batch_size):
        """
            产生DataLoader
        :param data: 原始的文本数据
        :param data_emb: 嵌入的数据
        :param batch_size: 批次大小
        :return: DataLoader
        """
        data["text"] = data["text"].apply(self.__english_f)
        data = data[["text", "label"]]

        data_emb["text"] = data_emb["text"].apply(self.__embedding_f)
        data_emb = data_emb[["text"]]

        my_data = MyDataset(data, data_emb)
        data_loader = DataLoader(my_data, batch_size=batch_size, shuffle=True)
        return data_loader

    def __loss_optim(self, model, lr, max_norm):
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=max_norm)

    def train_model(self, args, model):
        # 获取数据
        train_data = pd.read_csv(self.train_path, encoding='utf-8', header=0)
        test_data = pd.read_csv(self.val_path, encoding='utf-8', header=0)
        train_emb_data = pd.read_csv(self.emb_train_path, encoding='utf-8', header=0)
        val_emb_data = pd.read_csv(self.emb_val_path, encoding='utf-8', header=0)

        # 产生迭代器
        train_loader = self.__generate_data_loader(train_data, train_emb_data, args.batch_size)
        test_loader = self.__generate_data_loader(test_data, val_emb_data, len(test_data))

        # 产生损失函数和优化器
        self.__loss_optim(model, args.lr, args.max_norm)

        # 迭代训练
        steps = 0
        best_acc = 0
        last_step = 0
        model.train()
        for epoch in range(1, args.epochs + 1):
            for i, (feature, feature_emb, target) in enumerate(train_loader):
                if args.cuda:
                    feature, feature_emb, target = feature.cuda(), feature_emb.cuda(), target.cuda()
                self.optimizer.zero_grad()  # 梯度清零
                logits = model(feature, feature_emb)  # 前向传播
                loss = F.cross_entropy(logits, target)
                loss.backward()
                self.optimizer.step()
                steps += 1
                if steps % args.log_interval == 0:
                    corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                    train_acc = 100.0 * corrects / len(feature)
                    if len(feature) == args.batch_size:
                        sys.stdout.write(
                            '\rEpoch[{}] - Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,
                                                                                                 steps,
                                                                                                 loss.item(),
                                                                                                 train_acc,
                                                                                                 corrects,
                                                                                                 len(feature)))
                if steps % args.test_interval == 0:
                    result = self.__eval(test_loader, model, args)
                    if len(result) > 1:
                        dev_acc, y_true, y_pred = result[0], result[1], result[2]
                    else:
                        dev_acc, y_true, y_pred = result, None, None

                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        last_step = steps
                        if args.save_best:
                            print('Saving best model, acc: {:.4f}%.'.format(best_acc))
                            self.__save(model, args.save_dir, 'best', steps)
                        if args.confusion_matrix:
                            print('Confusion matrix:')
                            print(confusion_matrix(y_true, y_pred))
                        if args.show_score:
                            print('Score:')
                            accuracy = sm.accuracy_score(y_true, y_pred) * 100
                            precision = sm.precision_score(y_true, y_pred, average=args.average_type) * 100
                            recall = sm.recall_score(y_true, y_pred, average=args.average_type) * 100
                            f1_score = sm.f1_score(y_true, y_pred, average=args.average_type) * 100
                            auc = sm.roc_auc_score(y_true, y_pred, average=args.average_type) * 100
                            print('accuracy={:.2f}%.\n'
                                  'precision={:.2f}%.\n'
                                  'recall={:.2f}%.\n'
                                  'f1_score={:.2f}%.\n'
                                  'auc={:.2f}%.'.format(accuracy, precision, recall, f1_score, auc)
                                  )
                        if args.show_report:
                            print('Report:')
                            print(sm.classification_report(y_true, y_pred))
                    else:
                        if steps - last_step >= args.early_stopping:
                            print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                            raise KeyboardInterrupt

    @staticmethod
    def __eval(data_iter, model, args):
        model.eval()
        corrects, avg_loss = 0, 0
        for i, (feature, feature_emb, target) in enumerate(data_iter):
            if args.cuda:
                feature, feature_emb, target = feature.cuda(), feature_emb.cuda(), target.cuda()
            logits = model(feature, feature_emb)
            loss = F.cross_entropy(logits, target)
            avg_loss += loss.item()
            corrects += (torch.max(logits, 1)
                         [1].view(target.size()).data == target.data).sum()
            y_pred = torch.max(logits, 1)[1].view(target.size()).data
            y_true = target.data
            size = len(data_iter.dataset)
            avg_loss /= size
            accuracy = 100.0 * corrects / size
            print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                               accuracy,
                                                                               corrects,
                                                                               size))
            if args.confusion_matrix:
                return accuracy, y_true, y_pred
            else:
                return accuracy

    @staticmethod
    def __save(model, save_dir, save_prefix, steps):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
        torch.save(model.state_dict(), save_path)


def generate_user_embedding(path1, path2, path3, split_path=None, e_p=0.00001):
    '''
        产生数据集对应的微博关系嵌入
    :param e_p: 没有关系节点的嵌入
    :param path1: 微博关系的嵌入字典，json格式
    :param path2: 产生数据集对应的微博关系嵌入存储的地址
    :param path3: 原始的数据集地址
    :param split_path: 划分数据的路径，如果传入路径则对数据进行划分，否则为不做操作(None)。default：None
    :return: None
    '''
    with open(path1, 'r', encoding='UTF-8') as f:
        emb_dict = dict(json.loads(f.read()))  # 嵌入字典
        f.close()
    ori_data = pd.read_csv(path3, encoding='utf-8', header=0)  # 原始数据集对应的数据
    num_data = len(ori_data)  # 原始数据集大小
    emb_dim = 128  # 嵌入维度，训练关系嵌入时候的维度
    none_embedding = [e_p]*emb_dim   # 0.00001
    aim_data = {
        'text': [],  # 为了与后续数据集有对应，将其从'embedding'改为'text'，这边的'text'就是'embedding'
        'label': []
    }  # 待存储的数据
    for key in range(num_data):
        aim_data['label'].append(ori_data['label'][key])
        try:
            aim_data['text'].append(str(emb_dict[key]))
        except KeyError:
            aim_data['text'].append(str(none_embedding))
    aim_data = pd.DataFrame(aim_data)
    aim_data.to_csv(path2, encoding='utf-8', header=True, index=False)  # 存储数据
    # 划分数据
    if not split_path:
        pass
    divided_data(all_data=aim_data, split_path=split_path, divided_type='StratifiedKFold')


def divided_data(all_data, split_path, divided_type='train_test_split'):
    '''
        完成数据的划分
    :param all_data: 待划分的数据
    :param split_path: 划分数据存储的文件夹
    :param divided_type: 划分数据的类型 [train_test_split, KFold, StratifiedKFold]
    :return: None
    '''
    all_data = pd.DataFrame(all_data)
    dir_path = split_path + str(divided_type) + '/'
    if divided_type == 'train_test_split':
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        train, val = train_test_split(all_data, test_size=0.2, shuffle=True, random_state=20)
        train.to_csv(dir_path + "/train.csv", sep=',', index=False, encoding='utf-8', header=True)
        val.to_csv(dir_path + "/val.csv",  sep=',', index=False, encoding='utf-8', header=True)
    elif divided_type == 'KFold':
        kf = KFold(n_splits=5, shuffle=True, random_state=20)
        index_k = 1
        for train_index, val_index in kf.split(X=all_data):
            train, val = all_data.iloc[train_index], all_data.iloc[val_index]
            k_path = dir_path + str(index_k) + '/'
            if not os.path.isdir(k_path):
                os.makedirs(k_path)
            train.to_csv(k_path + "/train.csv", sep=',', index=False, encoding='utf-8', header=True)
            val.to_csv(k_path + "/val.csv", sep=',', index=False, encoding='utf-8', header=True)
            index_k += 1
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
        X = all_data
        Y = all_data['label']
        index_k = 1
        for train_index, val_index in skf.split(X, Y):
            train, val = all_data.iloc[train_index], all_data.iloc[val_index]
            k_path = dir_path + str(index_k) + '/'
            if not os.path.isdir(k_path):
                os.makedirs(k_path)
            train.to_csv(k_path + "/train.csv", sep=',', index=False, encoding='utf-8', header=True)
            val.to_csv(k_path + "/val.csv", sep=',', index=False, encoding='utf-8', header=True)
            index_k += 1
    print('Data have been divided.')


# data_set_name = 'OMD'  # 'HCR' 'OMD'
# generate_user_embedding(path1='./data/' + str(data_set_name) + '/relation_embedding_data/embedding.json',
#                         path2='./data/' + str(data_set_name) + '/relation_embedding_data/processed_data.csv',
#                         path3='./data/' + str(data_set_name) + '/processed_data.csv',
#                         split_path='./data/' + str(data_set_name) + '/relation_embedding_data/',
#                         e_p=0.00001)











