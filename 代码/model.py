import torch
import torch.nn as nn
import torch.nn.functional as F
# from base.base_model import BaseModel
from math import sqrt
from transformers import BertTokenizer, BertModel
import numpy as np


class TextCNN1d(nn.Module):
    '''
        TextCNN1D(nn.Module)
    '''
    def __init__(self, args):
        super(TextCNN1d, self).__init__()

        class_num = args.class_num
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        dropout = args.dropout
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes
        non_static = args.non_static
        static = args.static
        
        if static:
            self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(
                embeddings=args.vectors, freeze=non_static)
        else:
            self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dimension, out_channels=filter_num, kernel_size=fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dimension)
        x = self.embedding(x).float()
        # (batch_size, seq_len, embedding_dimension) -> (batch_size, embedding_dimension, seq_len)
        x = x.permute(0, 2, 1)
        # (batch_size, embedding_dimension, seq_len) -> [(batch_size, embedding_dimension, ?), ..., ...]
        x = [F.relu(conv(x)) for conv in self.convs]
        # [(batch_size, filter_num, ?), ..., ...] ->  [(batch_size, filter_num), ..., ...]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        # [(batch_size, filter_num), ..., ...] -> (batch_size, filter_num*len(filter_sizes))
        x = torch.cat(x, 1)
        # (batch_size, filter_num*len(filter_sizes)) -> (batch_size, filter_num*len(filter_sizes))
        x = self.dropout(x)
        # (batch_size, filter_num*len(filter_sizes)) -> (batch_size, class_num)
        logits = self.fc(x)
        return logits


class TextCNN(nn.Module):
    '''
        论文地址：https://arxiv.org/abs/1408.5882
        代码地址：https://github.com/bigboNed3/chinese_text_cnn
    '''
    def __init__(self, args):
        super(TextCNN, self).__init__()

        class_num = args.class_num
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        dropout = args.dropout
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes
        non_static = args.non_static
        static = args.static
        chanel_num = 1
        multichannel = args.multichannel

        if static:
            self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(
                embeddings=args.vectors, freeze=non_static)
        else:
            self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        if multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension)
            chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=chanel_num, out_channels=filter_num, kernel_size=(size, embedding_dimension)) for
             size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        if self.embedding2:
            # (batch_size, seq_len) -> (batch_size, 2, seq_len, embedding_dimension)
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dimension)
            x = self.embedding(x)
            # (batch_size, seq_len, embedding_dimension) -> (batch_size, 1, seq_len, embedding_dimension)
            x = x.unsqueeze(1)
        # (batch_size, chanel_num, seq_len, embedding_dimension) -> [(batch_size, filter_num, ?), ...]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # [(batch_size, filter_num, ?), ...] -> [(batch_size, filter_num), ...]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        # [(batch_size, filter_num), ...] -> (batch_size, filter_num*len(filter_sizes))
        x = torch.cat(x, 1)
        # (batch_size, filter_num*len(filter_sizes)) -> (batch_size, filter_num*len(filter_sizes))
        x = self.dropout(x)
        # # (batch_size, filter_num*len(filter_sizes)) -> (batch_size, class_num)
        logits = self.fc(x)
        return logits


class MLP(nn.Module):
    '''
        MLP: 多层感知器（取句子所有单词的平均作为词向量）
    '''
    def __init__(self, args):
        super().__init__()

        self.args = args

        class_num = args.class_num

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        hidden_dim = args.hidden_dim
        dropout = args.dropout

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, class_num),
        )

    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dimension)
        x = self.embedding(x)
        # (batch_size, seq_len, embedding_dimension) -> (batch_size, embedding_dimension)
        x = torch.mean(x, dim=1)
        # (batch_size, embedding_dimension) -> (batch_size, class_num)
        logits = self.mlp(x)
        return logits


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()

        self.args = args

        self.rnn_type = args.rnn_type.lower()
        bidirectional = args.bidirectional
        self.hidden_size = args.hidden_dim
        self.num_layers = args.num_layers
        class_num = args.class_num

        dropout = args.dropout
        embedding_dimension = args.embedding_dim
        vocabulary_size = args.vocabulary_size

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dimension,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout,
                               bias=True,
                               batch_first=False)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dimension,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              bias=True,
                              batch_first=True)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_dimension,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              bias=True,
                              batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(in_features=self.hidden_size*2, out_features=class_num)
        else:
            self.fc = nn.Linear(in_features=self.hidden_size, out_features=class_num)
        self.dropout = nn.Dropout(dropout)
        # self.hidden = self.init_hidden(batch_size)

    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dimension)
        x = self.embedding(x)
        # (batch_size, seq_len, embedding_dimension) -> (seq_len, batch_size, embedding_dimension)
        x = x.permute(1, 0, 2)
        if self.rnn_type == 'lstm':
            # (seq_len, batch_size, embedding_dimension) ->
            # (seq_len, batch_size, D*hidden_size),  # D = 2 if bidirectional else 1
            # ((num_layers*D, batch_size, hidden_size), (num_layers*D, batch_size, hidden_size))
            lstm_out, (hidden, cell) = self.rnn(x)
            # cat((batch_size, hidden_size), (batch_size, hidden_size)) -> (batch_size, hidden_size*2) # 最后的输出，ht和ht-1
            h = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            logits = self.fc(h)
        elif self.rnn_type in ['gru', 'rnn']:
            # (seq_len, batch_size, embedding_dimension) ->
            # (seq_len, batch_size, D*hidden_size),  # D = 2 if bidirectional else 1
            # (num_layers*D, batch_size, hidden_size)
            r_out, h_n = self.rnn(x, None)   # None 表示初始的 hidden state 为0
            # (seq_len, batch_size, D*hidden_size) -> batch_size, D*hidden_size) # 选取最后一个时间点的out输出
            drop_out = self.dropout(r_out[-1, :, :])
            # batch_size, D * hidden_size) -> ..
            logits = self.fc(drop_out)
            # batch_size, D * hidden_size) -> ..
            logits = nn.Softmax(dim=1)(logits)
        else:
            print('not have %s, please select in ["LSTM", "GRU", "RNN"]' % (self.rnn_type,))
            raise IndexError
        return logits

    def init_hidden(self, batch_size):
        """
        初始化隐状态：第一次送给LSTM时，没有隐状态，所以要初始化一个
        这里的初始化策略是全部赋0。
        这里之所以是tuple，是因为LSTM需要接受两个隐状态hidden state和cell state
        """
        hidden = (torch.zeros(self.num_layers*2, batch_size, self.hidden_size),
                  torch.zeros(self.num_layers*2, batch_size, self.hidden_size)
                  )
        return hidden


class RNNAttention(nn.Module):
    '''
        https://github.com/zy1996code/nlp_basic_model/blob/master/lstm_attention.py
    '''
    def __init__(self, args):
        super(RNNAttention, self).__init__()

        self.rnn_type = args.rnn_type.lower()
        bidirectional = args.bidirectional
        self.hidden_size = args.hidden_dim
        self.num_layers = args.num_layers
        class_num = args.class_num
        dropout = args.dropout

        embedding_dimension = args.embedding_dim
        vocabulary_size = args.vocabulary_size

        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.dropout = nn.Dropout(args.dropout)

        if self.num_layers == 2:  # 注意力机制中最好这麽用
            self.num_layers = 1
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dimension,
                               hidden_size=self.hidden_size,
                               bidirectional=bidirectional,
                               num_layers=self.num_layers,
                               dropout=dropout,
                               bias=True,
                               batch_first=False)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dimension,
                              hidden_size=self.hidden_size,
                              bidirectional=bidirectional,
                              num_layers=self.num_layers,
                              dropout=dropout,
                              bias=True,
                              batch_first=False)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_dimension,
                              hidden_size=self.hidden_size,
                              bidirectional=bidirectional,
                              num_layers=self.num_layers,
                              dropout=dropout,
                              bias=True,
                              batch_first=False)
        self.out = nn.Linear(self.hidden_size * self.D, class_num)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_size*self.D, self.num_layers)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, x):
        input = self.embedding(x)  # input : [batch_size, len_seq, embedding_dim]
        input = self.dropout(input)
        input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        hidden_state = torch.autograd.Variable(torch.zeros(self.num_layers*self.D, len(x), self.hidden_size)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.autograd.Variable(torch.zeros(self.num_layers*self.D, len(x), self.hidden_size)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        if self.rnn_type == 'lstm':
            output, (final_hidden_state, final_cell_state) = self.rnn(input, (hidden_state, cell_state))
        elif self.rnn_type in ['gru', 'rnn']:
            output, final_hidden_state = self.rnn(input, hidden_state)
        else:
            print('not have %s, please select in ["LSTM", "GRU", "RNN"]' % (self.rnn_type,))
            raise IndexError
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output)  # , attention  # model : [batch_size, num_classes], attention : [batch_size, n_step]


class TextRCNN(nn.Module):
    '''
        参考网址：https://blog.csdn.net/chen_yiwei/article/details/88598501?spm=1001.2014.3001.5502
        论文：Recurrent Convolutional Neural Networks for Text Classification
    '''
    def __init__(self, args):
        super(TextRCNN, self).__init__()

        self.args = args

        self.rnn_type = args.rnn_type.lower()
        bidirectional = args.bidirectional
        self.hidden_size = args.hidden_dim
        self.num_layers = args.num_layers
        class_num = args.class_num
        batch_size = args.batch_size

        dropout = args.dropout
        embedding_dimension = args.embedding_dim
        vocabulary_size = args.vocabulary_size
        self.device = args.device

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dimension,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout,
                               bias=True,
                               batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dimension,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              bias=True,
                              batch_first=True)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_dimension,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              bias=True,
                              batch_first=True)
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.fc = nn.Linear(in_features=self.hidden_size*2, out_features=class_num)
        else:
            self.fc = nn.Linear(in_features=self.hidden_size, out_features=class_num)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        if self.rnn_type == 'lstm':
            output, (final_hidden_state, final_cell_state) = self.rnn(x)
            output = torch.transpose(output, 1, 2)
            output = torch.tanh(output)
            output = F.max_pool1d(output, output.size(2))
            output = output.squeeze(2)
            logits = self.fc(output)
        elif self.rnn_type in ['gru', 'rnn']:
            r_out, h_n = self.rnn(x, None)
            drop_out = self.dropout(r_out[:, -1, :])
            logits = self.fc(drop_out)
        else:
            print('not have %s, please select in ["LSTM", "GRU", "RNN"]' % (self.rnn_type,))
            raise IndexError
        return logits


class FastText(nn.Module):
    def __init__(self, args):
        super(FastText, self).__init__()

        self.args = args

        self.hidden_size = args.hidden_dim
        class_num = args.class_num

        embedding_dimension = args.embedding_dim
        vocabulary_size = args.vocabulary_size
        self.device = args.device

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.fc1 = nn.Linear(embedding_dimension, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, class_num)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        h = self.fc1(x.mean(1))
        logits = self.fc2(h)
        logits = self.softmax(logits)
        return logits


class CNNRNN(nn.Module):
    '''
        一个多通道的CNNRNN模型，参考至师兄文章;
    '''
    def __init__(self, args):
        super(CNNRNN, self).__init__()

        self.args = args
        self.rnn_type = args.rnn_type.lower()
        self.hidden_size = args.hidden_dim
        multi_channel_num = args.multi_channel_num
        dropout = args.dropout
        filter_num = args.filter_num
        self.num_layers = args.num_layers
        bidirectional = args.bidirectional
        self.class_num = args.class_num

        self.embedding_dimension = args.embedding_dim
        vocabulary_size = args.vocabulary_size

        self.embedding = nn.Embedding(vocabulary_size, self.embedding_dimension)
        self.word_cnn = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embedding_dimension, out_channels=filter_num, kernel_size=(fs,))
             for fs in range(1, 1 + multi_channel_num)]
        )
        self.l_relu = nn.LeakyReLU(0.15)
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embedding_dimension,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout,
                               bias=True,
                               batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embedding_dimension,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              bias=True,
                              batch_first=True)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.embedding_dimension,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              bias=True,
                              batch_first=True)

    def selfatt(self, x, dim_q, dim_k, dim_v):
        q = nn.Linear(dim_q, dim_k, bias=False)(x)
        k = nn.Linear(dim_q, dim_k, bias=False)(x)
        v = nn.Linear(dim_q, dim_v, bias=False)(x)
        _norm_fact = 1 / sqrt(dim_k)

        dist = torch.bmm(q, k.transpose(1, 2)) * _norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.bmm(dist, v)
        return att

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, len_s) -> (batch_size, len_s, word_dim)
        dim_k = x.size()[1]
        x = self.dropout(x)  # (batch_size, len_s, word_dim)
        x = x.permute(0, 2, 1)  # (batch_size, word_dim, len_s)
        x = [F.relu(conv(x)) for conv in self.word_cnn]
        # x = [F.leaky_relu(conv(x), negative_slope=0.15) for conv in self.word_cnn]
        x = [F.dropout(item) for item in x]
        x = [item.permute(0, 2, 1) for item in x]

        rnn_out_list = []
        for item in x:
            hidden_state = torch.autograd.Variable(torch.zeros(self.num_layers * self.D, len(item),
                                                               self.hidden_size))
            cell_state = torch.autograd.Variable(torch.zeros(self.num_layers * self.D, len(item),
                                                             self.hidden_size))
            if self.rnn_type == 'lstm':
                output, (final_hidden_state, final_cell_state) = self.rnn(item, (hidden_state, cell_state))
            elif self.rnn_type in ['gru', 'rnn']:
                output, final_hidden_state = self.rnn(item, hidden_state)
            else:
                print('not have %s, please select in ["LSTM", "GRU", "RNN"]' % (self.rnn_type,))
                raise IndexError
            output = self.selfatt(output, output.size()[2], dim_k, output.size()[2])
            output = self.dropout(output)
            rnn_out_list.append(output)
        x = torch.cat(rnn_out_list, 1)
        x = self.dropout(x)
        x = torch.reshape(x, (x.size()[0], x.size()[1]*x.size()[2]))
        # x = nn.BatchNorm1d(x.size()[1], affine=False)(x)  # 加上有可能报错：https://blog.csdn.net/qq_45365214/article/details/122670591
        x = nn.Linear(x.size()[1], self.hidden_size)(x)
        x = nn.Linear(self.hidden_size, self.hidden_size)(x)
        x = nn.Linear(self.hidden_size, self.hidden_size)(x)
        x = nn.Linear(self.hidden_size, self.hidden_size)(x)
        logits = nn.Linear(x.size()[1], self.class_num)(x)
        return logits


class Transformer(nn.Module):
    '''
        Transform
        原理：https://zhuanlan.zhihu.com/p/338817680
        代码：http://nlp.seas.harvard.edu/2018/04/03/attention.html
             https://gitcode.net/mirrors/kyubyong/transformer?utm_source=csdn_github_accelerator
        解释：Transformer是第一个用纯attention搭建的模型，不仅计算速度更快，在翻译任务上也获得了更好的结果。Google现在的翻译应该是在此基础上做的，但是请教了一两个朋友，得到的答案是主要看数据量，数据量大可能用transformer好一些，小的话还是继续用rnn-based model
        解释来源：https://blog.csdn.net/SMith7412/article/details/88755019
        paper：https://arxiv.org/abs/1706.03762
    '''
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.args = args
        self.rnn_type = args.rnn_type.lower()
        self.hidden_size = args.hidden_dim
        multi_channel_num = args.multi_channel_num
        dropout = args.dropout
        filter_num = args.filter_num
        self.num_layers = args.num_layers
        bidirectional = args.bidirectional
        self.class_num = args.class_num

        self.embedding_dimension = args.embedding_dim
        vocabulary_size = args.vocabulary_size

        self.embedding = nn.Embedding(vocabulary_size, self.embedding_dimension)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dimension,
                                                        nhead=8,
                                                        dim_feedforward=2048,
                                                        dropout=0.1,
                                                        activation='relu',  # "relu" or "gelu"
                                                        layer_norm_eps=1e-5,
                                                        batch_first=False,
                                                        norm_first=False,  # after
                                                        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                         num_layers=2,
                                                         norm=None
                                                         )
        self.fc = nn.Linear(self.embedding_dimension, self.class_num)

    def forward(self, x):
        src_x = self.embedding(x)
        out_x = self.transformer_encoder(src_x)
        mean_x = torch.mean(out_x, dim=1)
        logits = self.fc(mean_x)
        return logits


class BERT(nn.Module):
    """
        https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch/blob/master/models/bert.py
    """
    def __init__(self, args):
        super(BERT, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.cls_layer = nn.Linear(bert_input_size, class_num)
        self.out = nn.Linear(bert_input_size, bert_input_size)
        self.out1 = nn.Linear(bert_input_size, bert_input_size)
        self.out2 = nn.Linear(bert_input_size, bert_input_size)

    def forward(self, x):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            # last_hidden_states[0].shape = (batch_size len_sql, input_size)
            # last_hidden_states[0][:, 0] last_hidden_states[0][:][0][:]
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']
        x = self.out(bert_output)
        x = self.out1(x)
        x = self.out2(x)
        logits = self.cls_layer(x)
        return logits


class BERTCNN(nn.Module):
    '''
        https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch/blob/master/models/bert_CNN.py
    '''
    def __init__(self, args):
        super(BERTCNN, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num
        filter_sizes = args.filter_sizes
        filter_num = args.filter_num
        dropout = args.dropout

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (size, bert_input_size)) for size in filter_sizes]
        )

        self.dropout = nn.Dropout(dropout)
        self.cls_layer = nn.Linear(filter_num * len(filter_sizes), class_num)

    def forward(self, x):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']
        x = last_hidden_state.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.cls_layer(x)
        return logits


class BERTDPCNN(nn.Module):
    """
        name:Deep Pyramid Convolutional Neural Networks for Text Categorization(DPCNN)
        原因：由于TextCNN不能通过卷积获得文本的长距离关系依赖，而论文中DPCNN通过不断加深网络，可以抽取长距离的文本依赖关系
        知乎：https://zhuanlan.zhihu.com/p/372904980
        论文：https://aclanthology.org/P17-1052/
    """
    def __init__(self, args):
        super(BERTDPCNN, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num
        filter_sizes = args.filter_sizes
        filter_num = args.filter_num
        dropout = args.dropout

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.conv_region = nn.Conv2d(1, filter_num, (3, bert_input_size), stride=1)
        self.conv = nn.Conv2d(filter_num, filter_num, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom (左右上下)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(filter_num, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']

        # TODO DPCNN
        # (batch_size, seq_len, embed) -> (batch_size, 1, seq_len, embed)
        x = last_hidden_state.unsqueeze(1)
        # (batch_size, 1, seq_len, embed) -> (batch_size, filter_num, seq_len-3+1, 1)
        x = self.conv_region(x)
        # (batch_size, filter_num, seq_len-3+1, 1) -> (batch_size, filter_num, seq_len, 1)
        x = self.padding1(x)
        # (batch_size, filter_num, seq_len, 1) -> ..
        x = self.relu(x)
        # (batch_size, filter_num, seq_len, 1) -> (batch_size, filter_num, seq_len-3+1, 1)
        x = self.conv(x)
        # (batch_size, filter_num, seq_len-3+1, 1) -> (batch_size, filter_num, seq_len, 1)
        x = self.padding1(x)
        # (batch_size, filter_num, seq_len, 1) -> ..
        x = self.relu(x)
        # (batch_size, filter_num, seq_len, 1) -> (batch_size, filter_num, seq_len-3+1, 1)
        x = self.conv(x)
        while x.size()[2] >= 2:
            x = self._block(x)
        # (batch_size, filter_num, 1, 1) -> (batch_size, filter_num, 1)
        x = x.squeeze(dim=2)
        # (batch_size, filter_num, 1) -> (batch_size, filter_num)
        x = x.squeeze(dim=2)
        # (batch_size, filter_num) -> (batch_size, filter_num)
        x = self.fc(x)
        # (batch_size, filter_num) -> (batch_size, filter_num)
        logits = self.softmax(x)
        return logits

    def _block(self, x):
        # (batch_size, filter_num, seq_len-3+1, 1) -> (batch_size, filter_num, seq_len-1, 1)
        x = self.padding2(x)
        # (batch_size, filter_num, seq_len-1, 1) -> (batch_size, filter_num, ?, 1)
        px = self.max_pool(x)
        # (batch_size, filter_num, ?) -> (batch_size, filter_num, ?+2, 1)
        x = self.padding1(px)
        # (batch_size, filter_num, ?+2, 1) -> ..
        x = F.relu(x)
        # (batch_size, filter_num, ?, 1) -> (batch_size, filter_num, ?-3+1, 1)
        x = self.conv(x)
        # (batch_size, filter_num, ?, 1) -> (batch_size, filter_num, ?+3-1, 1)
        x = self.padding1(x)
        # (batch_size, filter_num, ?+3-1, 1) -> (batch_size, filter_num, ?+3-1, 1)
        x = F.relu(x)
        # (batch_size, filter_num, ?, 1) -> (batch_size, filter_num, ?-3+1, 1)
        x = self.conv(x)
        x = x + px  # short cut
        return x


class TextRNN(torch.nn.Module):
    """
        https://blog.csdn.net/chen_yiwei/article/details/88598501?spm=1001.2014.3001.5502
    """
    def __init__(self, args):
        super(TextRNN, self).__init__()

        class_num = args.class_num
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        static = args.static
        dropout = args.dropout
        num_layers = args.num_layers
        hidden_dim = args.hidden_dim
        non_static = args.non_static
        bidirectional = args.bidirectional
        if static:
            self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(
                embeddings=args.vectors, freeze=non_static)
        else:
            self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        self.lstm = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.embed_dropout = nn.Dropout(dropout)
        if bidirectional:
            D = 2
        else:
            D = 1
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * D * num_layers, hidden_dim * D * num_layers),
            nn.ReLU(),
            nn.Linear(hidden_dim * D * num_layers, hidden_dim * D * num_layers),
            nn.ReLU(),
            nn.Linear(hidden_dim * D * num_layers, hidden_dim * D * num_layers),
            nn.ReLU(),
            nn.Linear(hidden_dim * D * num_layers, class_num)
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        lstm_out = final_hidden_state.permute(1, 0, 2)
        lstm_out = lstm_out.contiguous().view(int(x.shape[0]), -1)
        out = self.mlp(lstm_out)
        out = self.softmax(out)
        return out


class BERTRCNN(torch.nn.Module):
    """
        参考：https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch/tree/master/models
    """
    def __init__(self, args):
        super(BERTRCNN, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num
        hidden_size = args.hidden_dim
        num_layers = args.num_layers
        dropout = args.dropout
        bidirectional = args.bidirectional

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.lstm = nn.LSTM(input_size=bert_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True,
                            bias=True)
        # self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(hidden_size*2 + bert_input_size, class_num)

    def forward(self, x):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']
        out, _ = self.lstm(last_hidden_state)
        out = torch.cat((last_hidden_state, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = nn.MaxPool1d(out[1])(out).squeeze()
        out = self.fc(out)
        return out


class BERTRNN(torch.nn.Module):
    """
        https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch/tree/master/models
    """
    def __init__(self, args):
        super(BERTRNN, self).__init__()

        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num
        hidden_size = args.hidden_dim
        num_layers = args.num_layers
        dropout = args.dropout
        bidirectional = args.bidirectional

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.lstm = nn.LSTM(input_size=bert_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True,
                            bias=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(hidden_size*2, class_num)

    def forward(self, x):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']
        out, _ = self.lstm(last_hidden_state)
        out = self.dropout(out)
        out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out


class ERNIE(nn.Module):
    """
        参考：https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch/tree/master/models
        论文：Enhanced Representation through kNowledge IntEgration
    """
    def __init__(self, args):
        super(ERNIE, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.cls_layer = nn.Linear(bert_input_size, class_num)

    def forward(self, x):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            # last_hidden_states[0].shape = (batch_size len_sql, input_size)
            # last_hidden_states[0][:, 0] last_hidden_states[0][:][0][:]
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']
        logits = self.cls_layer(bert_output)
        return logits


class BertCNNEmb(nn.Module):
    """
        自己的模型：基于BERT-CNN-Embedding
    """

    def __init__(self, args):
        super(BertCNNEmb, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num
        filter_sizes = args.filter_sizes
        filter_num = args.filter_num
        dropout = args.dropout
        rel_embedding_dim = args.rel_embedding_dim

        self.class_num = class_num
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (size, bert_input_size)) for size in filter_sizes]
        )

        # 卷积输出
        self.dropout = nn.Dropout(dropout)
        self.cls_conv = nn.Linear(filter_num * len(filter_sizes), rel_embedding_dim)

        # 嵌入输出
        self.dropout1 = nn.Dropout(dropout)

        # pooler输出
        self.dropout2 = nn.Dropout(dropout)
        self.cls_pooler = nn.Linear(bert_input_size, rel_embedding_dim)

        # 组合: torch.stack()
        self.switch = nn.Linear(3, 1, bias=False)  # True -> False

        # 输出
        self.cls_layer = nn.Linear(rel_embedding_dim, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_emb):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']
        x = last_hidden_state.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, dim=1)

        # 卷积输出
        x = self.dropout(x)
        x_conv_out = self.cls_conv(x)

        # 嵌入输出
        emb_out = self.dropout1(x_emb)

        # pooler输出
        bert_output = self.dropout2(bert_output)
        pooler_out = self.cls_pooler(bert_output)

        # 输出堆叠
        stack_out = torch.stack((x_conv_out, emb_out, pooler_out), dim=2)
        switch_out = self.switch(stack_out)

        # 输出分类
        cls = torch.squeeze(switch_out, dim=2)
        logits = self.cls_layer(cls)
        logits = self.softmax(logits)
        return logits


class BERTDPCNNEmb(nn.Module):
    """
        name:Deep Pyramid Convolutional Neural Networks for Text Categorization(DPCNN)
        原因：由于TextCNN不能通过卷积获得文本的长距离关系依赖，而论文中DPCNN通过不断加深网络，可以抽取长距离的文本依赖关系
        知乎：https://zhuanlan.zhihu.com/p/372904980
        论文：https://aclanthology.org/P17-1052/
    """
    def __init__(self, args):
        super(BERTDPCNNEmb, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num
        filter_sizes = args.filter_sizes
        filter_num = args.filter_num
        dropout = args.dropout
        rel_embedding_dim = args.rel_embedding_dim

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.conv_region = nn.Conv2d(1, filter_num, (3, bert_input_size), stride=1)
        self.conv = nn.Conv2d(filter_num, filter_num, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom (左右上下)
        self.relu = nn.ReLU()

        # 卷积输出
        self.dropout = nn.Dropout(dropout)
        self.cls_conv = nn.Linear(filter_num, rel_embedding_dim)

        # 嵌入输出
        self.dropout1 = nn.Dropout(dropout)

        # pooler输出
        self.dropout2 = nn.Dropout(dropout)
        self.cls_pooler = nn.Linear(bert_input_size, rel_embedding_dim)

        # 组合: torch.stack()
        self.switch = nn.Linear(3, 1, bias=True)

        # 输出
        self.cls_layer = nn.Linear(rel_embedding_dim, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_emb):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']

        # TODO DPCNN
        # (batch_size, seq_len, embed) -> (batch_size, 1, seq_len, embed)
        x = last_hidden_state.unsqueeze(1)
        # (batch_size, 1, seq_len, embed) -> (batch_size, filter_num, seq_len-3+1, 1)
        x = self.conv_region(x)
        # (batch_size, filter_num, seq_len-3+1, 1) -> (batch_size, filter_num, seq_len, 1)
        x = self.padding1(x)
        # (batch_size, filter_num, seq_len, 1) -> ..
        x = self.relu(x)
        # (batch_size, filter_num, seq_len, 1) -> (batch_size, filter_num, seq_len-3+1, 1)
        x = self.conv(x)
        # (batch_size, filter_num, seq_len-3+1, 1) -> (batch_size, filter_num, seq_len, 1)
        x = self.padding1(x)
        # (batch_size, filter_num, seq_len, 1) -> ..
        x = self.relu(x)
        # (batch_size, filter_num, seq_len, 1) -> (batch_size, filter_num, seq_len-3+1, 1)
        x = self.conv(x)
        while x.size()[2] >= 2:
            x = self._block(x)
        x = x.squeeze(dim=2)
        x = x.squeeze(dim=2)

        # 卷积输出
        x_conv_out = self.cls_conv(x)

        # 嵌入输出
        emb_out = self.dropout1(x_emb)

        # pooler输出
        bert_output = self.dropout2(bert_output)
        pooler_out = self.cls_pooler(bert_output)

        # 输出堆叠
        stack_out = torch.stack((x_conv_out, emb_out, pooler_out), dim=2)
        switch_out = self.switch(stack_out)

        # 输出分类
        cls = torch.squeeze(switch_out, dim=2)
        logits = self.cls_layer(cls)
        logits = self.softmax(logits)
        return logits

    def _block(self, x):
        # (batch_size, filter_num, seq_len-3+1, 1) -> (batch_size, filter_num, seq_len-1, 1)
        x = self.padding2(x)
        # (batch_size, filter_num, seq_len-1, 1) -> (batch_size, filter_num, ?, 1)
        px = self.max_pool(x)
        # (batch_size, filter_num, ?) -> (batch_size, filter_num, ?+2, 1)
        x = self.padding1(px)
        # (batch_size, filter_num, ?+2, 1) -> ..
        x = F.relu(x)
        # (batch_size, filter_num, ?, 1) -> (batch_size, filter_num, ?-3+1, 1)
        x = self.conv(x)
        # (batch_size, filter_num, ?, 1) -> (batch_size, filter_num, ?+3-1, 1)
        x = self.padding1(x)
        # (batch_size, filter_num, ?+3-1, 1) -> (batch_size, filter_num, ?+3-1, 1)
        x = F.relu(x)
        # (batch_size, filter_num, ?, 1) -> (batch_size, filter_num, ?-3+1, 1)
        x = self.conv(x)
        x = x + px  # short cut
        return x


class BertCNNEmbSim(nn.Module):
    """
        自己的模型：基于BERT-CNN-Embedding
    """

    def __init__(self, args):
        super(BertCNNEmbSim, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num
        filter_sizes = args.filter_sizes
        filter_num = args.filter_num
        dropout = args.dropout
        rel_embedding_dim = args.rel_embedding_dim

        # 双通道
        self.multichannel = args.multichannel  # !!!!!!!!!!!!!
        bert_path2 = args.bert_path2   # !!!!!!!!!!!!!
        bert_input_size2 = args.bert_input_size2
        if self.multichannel:
            channel_num = 2
        else:
            channel_num = 1
        self.tokenizer2 = BertTokenizer.from_pretrained(bert_path2)
        self.bert2 = BertModel.from_pretrained(bert_path2)
        # 映射
        self.bert2_to_1 = nn.Linear(bert_input_size2, bert_input_size)
        self.cls_pooler2 = nn.Linear(bert_input_size2, rel_embedding_dim)

        # rnn
        # bidirectional = args.bidirectional   # !!!!!!!!!!!!!
        #
        # self.filter_num_temp = filter_num  # !!!!!!!!!!!!!!!!!
        #
        # self.hidden_size = args.hidden_dim   # !!!!!!!!!!!!!
        # self.num_layers = args.num_layers  # !!!!!!!!!!!!!

        self.class_num = class_num
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, bert_input_size)) for size in filter_sizes]
        )

        # 卷积输出
        self.dropout = nn.Dropout(dropout)
        self.cls_conv = nn.Linear(filter_num * len(filter_sizes), rel_embedding_dim)

        # 嵌入输出
        self.dropout1 = nn.Dropout(dropout)

        # 社交嵌入转换
        # self.social_change = nn.Linear(rel_embedding_dim, filter_num * len(filter_sizes))

        # pooler输出
        self.dropout2 = nn.Dropout(dropout)
        self.cls_pooler = nn.Linear(bert_input_size, rel_embedding_dim)

        # 组合: 卷积
        filter_sizes2 = [1, 2, 3]
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (size, rel_embedding_dim)) for size in filter_sizes2]
        )

        # rnn
        # self.rnn = nn.RNN(input_size=filter_num,
        #                   hidden_size=self.hidden_size,
        #                   num_layers=self.num_layers,
        #                   bidirectional=bidirectional,
        #                   dropout=dropout,
        #                   bias=True,
        #                   batch_first=True)
        # if bidirectional:
        #     self.D = 2
        # else:
        #     self.D = 1

        # 输出
        self.cls_layer = nn.Linear(filter_num * len(filter_sizes2), class_num)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def selfatt(x_q, x_k, x_v):
        dim_q = x_q.size(1)
        dim_k = x_k.size(1)
        dim_v = x_v.size(1)
        assert dim_q == dim_k and dim_k == dim_v
        q = nn.Linear(dim_q, dim_k, bias=True)(x_q)
        k = nn.Linear(dim_q, dim_k, bias=True)(x_k)
        v = nn.Linear(dim_q, dim_v, bias=True)(x_v)
        _norm_fact = 1 / sqrt(dim_k)

        dist = torch.matmul(q, k.transpose(0, 1)) * _norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.matmul(dist, v)
        return att

    def forward(self, x, x_emb):
        tokens = self.tokenizer(x, padding=True)
        tokens2 = self.tokenizer2(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        input_ids2 = torch.tensor(tokens2["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        attention_mask2 = torch.tensor(tokens2["attention_mask"])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
            last_hidden_states2 = self.bert2(input_ids2, attention_mask=attention_mask2)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']
        last_hidden_state2, bert_output2 = last_hidden_states2['last_hidden_state'], last_hidden_states2['pooler_output']

        if self.multichannel:
            last_hidden_state2 = self.bert2_to_1(last_hidden_state2)  # bert 2 -> 1
            x = torch.stack([last_hidden_state, last_hidden_state2], dim=1)
        else:
            x = last_hidden_state.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, dim=1)

        # 卷积输出
        x = self.dropout(x)
        x_conv_out = self.cls_conv(x)

        # 嵌入输出
        emb_out = self.dropout1(x_emb)
        # emb_out = self.social_change(emb_out)
        # 嵌入去除噪声
        emb_n = torch.ones(int(emb_out.shape[0]), int(emb_out.shape[1]))*0.00001
        emb_s_n = torch.sum(emb_out - emb_n, dim=1).mul(torch.sum(emb_out - emb_n, dim=1)) * 10**15
        emb_s_n = F.tanh(emb_s_n)
        a = emb_s_n.reshape(int(emb_out.shape[0]), 1)
        b = emb_s_n.reshape(1, int(emb_out.shape[0]))
        xiuzheng_emb = F.relu(torch.tanh((torch.mm(a, b) - torch.eye(int(emb_out.shape[0]))) * 10))

        # pooler输出
        bert_output = self.dropout2(bert_output)
        pooler_out = self.cls_pooler(bert_output)
        # pooler_out2 = self.cls_pooler2(bert_output2)

        # 社交指导融合
        emb_out1 = emb_out / torch.norm(emb_out, dim=-1, keepdim=True)
        sim_emb_out = torch.mm(emb_out1, emb_out1.T) - xiuzheng_emb
        sim_emb_out = sim_emb_out / torch.norm(sim_emb_out, dim=-1, keepdim=True)

        sim_pooler_out = torch.matmul(sim_emb_out, pooler_out)
        sim_x_conv_out = torch.matmul(sim_emb_out, x_conv_out)

        # 卷积堆叠输出
        self_att_out = self.selfatt(x_q=emb_out, x_k=sim_pooler_out, x_v=sim_x_conv_out)  # emb_out 1

        stack_out = torch.stack((x_conv_out, pooler_out, self_att_out), dim=1)  # （B, 6, rel_embedding_dim）  # sim_x_conv_out, sim_pooler_out, emb_out1
        stack_out = stack_out.unsqueeze(1)
        stack_out = [F.relu(conv(stack_out)).squeeze(3) for conv in self.convs1]

        # stack_out = [item.permute(0, 2, 1) for item in stack_out]   # [(b, ., filter_num), ....]
        # stack_out = [self.rnn(item, None)[0][:, -1, :] for item in stack_out]  # [b, D*hidden_size, ......]
        # stack_out = [nn.Linear(self.D*self.hidden_size, self.filter_num_temp)(i) for i in stack_out]
        stack_out = [F.avg_pool1d(item, item.size(2)).squeeze(2) for item in stack_out]

        stack_out = torch.cat(stack_out, dim=1)

        # 输出分类
        logits = self.cls_layer(stack_out)
        logits = self.softmax(logits)
        return logits


# class BertCNNEmbSim(nn.Module):
#     """
#         自己的模型：基于BERT-CNN-Embedding
#     """
#
#     def __init__(self, args):
#         super(BertCNNEmbSim, self).__init__()
#         bert_path = args.bert_path
#         bert_input_size = args.bert_input_size
#         class_num = args.class_num
#         filter_sizes = args.filter_sizes
#         filter_num = args.filter_num
#         dropout = args.dropout
#         rel_embedding_dim = args.rel_embedding_dim
#
#         self.class_num = class_num
#         self.tokenizer = BertTokenizer.from_pretrained(bert_path)
#         self.bert = BertModel.from_pretrained(bert_path)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, filter_num, (size, bert_input_size)) for size in filter_sizes]
#         )
#
#         # 卷积输出
#         self.dropout = nn.Dropout(dropout)
#         self.cls_conv = nn.Linear(filter_num * len(filter_sizes), rel_embedding_dim)
#
#         # 嵌入输出
#         self.dropout1 = nn.Dropout(dropout)
#
#         # pooler输出
#         self.dropout2 = nn.Dropout(dropout)
#         self.cls_pooler = nn.Linear(bert_input_size, rel_embedding_dim)
#
#         # 组合: torch.stack()
#         self.switch = nn.Linear(4, 1, bias=False)  # 之前师True
#
#         # 输出
#         self.cls_layer = nn.Linear(rel_embedding_dim, class_num)
#         self.softmax = nn.Softmax(dim=1)
#
#     @staticmethod
#     def selfatt(x_q, x_k, x_v):
#         dim_q = x_q.size(1)
#         dim_k = x_k.size(1)
#         dim_v = x_v.size(1)
#         assert dim_q == dim_k and dim_k == dim_v
#         q = nn.Linear(dim_q, dim_k, bias=True)(x_q)
#         k = nn.Linear(dim_q, dim_k, bias=True)(x_k)
#         v = nn.Linear(dim_q, dim_v, bias=True)(x_v)
#         _norm_fact = 1 / sqrt(dim_k)
#
#         dist = torch.matmul(q, k.transpose(0, 1)) * _norm_fact
#         dist = torch.softmax(dist, dim=-1)
#         att = torch.matmul(dist, v)
#         return att
#
#     def forward(self, x, x_emb):
#         tokens = self.tokenizer(x, padding=True)
#         input_ids = torch.tensor(tokens["input_ids"])
#         attention_mask = torch.tensor(tokens["attention_mask"])
#         with torch.no_grad():
#             last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
#         last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states['pooler_output']
#         x = last_hidden_state.unsqueeze(1)
#         x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
#         x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
#         x = torch.cat(x, dim=1)
#
#         # 卷积输出
#         x = self.dropout(x)
#         x_conv_out = self.cls_conv(x)
#
#         # 嵌入输出
#         emb_out = self.dropout1(x_emb)
#
#         # pooler输出
#         bert_output = self.dropout2(bert_output)
#         pooler_out = self.cls_pooler(bert_output)
#
#         # 社交指导融合
#         sim_emb_out = torch.matmul(emb_out, emb_out.T) * (1 / sqrt(int(x.size(0))))
#         sim_pooler_out = torch.matmul(sim_emb_out, pooler_out)
#         pooler_out = torch.sub(sim_pooler_out, pooler_out)
#
#         sim_x_conv_out = torch.matmul(sim_emb_out, x_conv_out)
#         x_conv_out = torch.sub(x_conv_out, sim_x_conv_out)
#
#         # 输出堆叠
#         self_att_out = self.selfatt(x_q=x_conv_out, x_k=emb_out, x_v=pooler_out)
#         stack_out = torch.stack((x_conv_out, emb_out, pooler_out, self_att_out), dim=2)
#         switch_out = self.switch(stack_out)
#
#         # 输出分类
#         cls = torch.squeeze(switch_out, dim=2)
#         logits = self.cls_layer(cls)
#         logits = self.softmax(logits)
#         return logits

















































