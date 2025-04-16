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
'''
    https://github.com/huggingface/datasets/tree/master/datasets
    BERT的使用：https://zhuanlan.zhihu.com/p/502554832
    中文BERT的使用：https://github.com/ymcui/Chinese-BERT-wwm
    BERT预训练模型的使用:
    1. https://huggingface.co/ 这里下载模型(可以通过git下载，然后大文件直接点击下载)，下载的参考网址如下。只用下载config.json、
        pytorch_model.bin、vocab.txt 3个文件；
        https://blog.csdn.net/qq_39448884/article/details/123908752
        https://blog.csdn.net/WillWinston/article/details/120079464
        https://blog.csdn.net/weixin_43646592/article/details/119520963
        https://blog.csdn.net/weixin_41318625/article/details/123498867
    2. git安装可参考：git安装.txt
    3. transform 可以参考：
        https://zhuanlan.zhihu.com/p/396221959
    4. BERT预训练模型的解释：
        4.1 网址：
            https://zhuanlan.zhihu.com/p/469612346
            https://zhuanlan.zhihu.com/p/103226488?utm_source=wechat_session
            https://blog.csdn.net/stay_foolish12/article/details/112366097 # 
            https://blog.csdn.net/pearl8899/article/details/116354207  # 参数解释：
            https://huggingface.co/docs/transformers/model_doc/bert#bertmodellmheadmodel  # Hugging Face 解释
        4.2 参数解释：
            4.2.1 输入：
                input_ids: 必须有
                    类型：torch.LongTensor ，形状： (batch_size, sequence_length)
                    意义：输入文本在词典中的映射id，又称tokens
                attention_mask：optional，非必须
                    类型：torch.FloatTensor ，形状：(batch_size, sequence_length)
                    意义：避免对padding(填充，因为输入需要时定长，不够长度的，用0补齐)的tokens索引过于关注，所以这里用0和1做mask。0表示是padding产生的值；1表示原文，可有进行关注的词。
                token_type_ids：optional，非必须
                    类型：torch.LongTensor ，形状：(batch_size, sequence_length)
                    意义：用来区分一条样本中的第一部分和第二部分，同样用0和1来区分。0表示第一部分，1表示第二部分。常用在句子预测、问答场景下。
                position_ids： optional，非必须
                    类型：torch.LongTensor ，形状： (batch_size, sequence_length)
                    意义：在位置embedding中，每个输入序列tokens的位置索引。范围[0,config.max_position_embeddings - 1]。这个目前没有用到，没有研究。
            4.2.1 输出：
                last_hidden_state：
                    类型：torch.FloatTensor ，形状： (batch_size, sequence_length, hidden_size)
                    意义：模型最后一层输出的隐藏层状态hidden_state序列。记录了整个文本的计算之后的每一个 token 的结果信息。
                pooler_out：
                    类型：torch.FloatTensor，形状 (batch_size, hidden_size)
                    意义：最后一层由线性层和Tanh激活函数进一步处理过的序列的第一个token(分类token)的隐藏状态。在预训练过程中，线性层权值会根据你的任务进行参数更新。
                        代表序列的第一个 token 的最后一个隐藏层的状态。shape 是 (batch_size, hidden_size)。所谓的第一个 token，就是咱们刚才提到的[CLS]标签。
                        可以作为文本的embedding。bert系列的其他类，没有这个输出选项。
                        1].Token embeddings：词向量。这里需要注意的是，Token embeddings 的第一个开头的 token 一定得是“[CLS]”。[CLS]作为整篇文本的语义表示，用于文本分类等任务。
                        2].Segment embeddings。这个向量主要是用来将两句话进行区分，比如问答任务，会有问句和答句同时输入，这就需要一个能够区分两句话的操作。不过在咱们此次的分类任务中，只有一个句子。
                        3].Position embeddings。记录了单词的位置信息。
'''


# 数据集
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


class TrainBert(object):
    """
        训练BERT模型:
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    def __init__(self, data_set_name, k):
        self.regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')  # 去除's、@、#、
        # regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
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

    def __english_f(self, text):
        # text = " ".join([i for i in text.split() if (i[:4] != 'http') and (i[:1] != '#') and (i[:1] != '@')])
        text = self.regex.sub(' ', text)
        text = " ".join([word.lower() for word in text.split() if (word.strip()) and (word.lower() not in self.stopwords)])
        # text = " ".join(
        #     [word.lower() for word in text.split() if (word.strip()) and (word.lower() not in self.stopwords) and
        #      (word[0] != '#') and (word[0] != '@')])
        return text

    def __generate_data_loader(self, data, batch_size):
        data["text"] = data["text"].apply(self.__english_f)
        data = data[["text", "label"]]
        my_data = MyDataset(data)
        data_loader = DataLoader(my_data, batch_size=batch_size, shuffle=True)
        return data_loader

    def __loss_optim(self, model, lr):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.criterion = nn.BCELoss()
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate, last_epoch=-1, verbose=True)
        # https://blog.csdn.net/weixin_45209433/article/details/112324325

    def train_model(self, args, model):
        # 划分数据
        train_data = pd.read_csv(self.train_path, encoding='utf-8', header=0)
        test_data = pd.read_csv(self.val_path, encoding='utf-8', header=0)
        # 产生迭代器
        train_loader = self.__generate_data_loader(train_data, args.batch_size)
        test_loader = self.__generate_data_loader(test_data, len(test_data))
        # 产生损失函数和优化器
        self.__loss_optim(model, args.lr)
        # 迭代训练
        steps = 0
        best_acc = 0
        last_step = 0
        model.train()
        for epoch in range(1, args.epochs + 1):
            for i, (feature, target) in enumerate(train_loader):
                if args.cuda:
                    feature, target = feature.cuda(), target.cuda()
                self.optimizer.zero_grad()  # 梯度清零
                logits = model(feature)  # 前向传播
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
        for i, (feature, target) in enumerate(data_iter):
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            logits = model(feature)
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







