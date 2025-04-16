import argparse


MODEL_NAMES = ['TextCNN1d', "TextCNN", 'BERT', "BERTCNN", "MLP", "RNN", "RNNAttention", "RCNN", "FastText", "CNNRNN",
               "Transformer", "BertCNNEmb", "BERTDPCNN", "BERTDPCNNEmb", "BertCNNEmbSim", "TextRNN",  "BERT_ERNIE"]

DATASET_NAMES = ['OMD', 'HCR']
DATASRT_NAME = DATASET_NAMES[0]
MODEL_NAME = MODEL_NAMES[-3]  # -4
NEED_WORD2VEC = False


def config(par=None):
    parser = par

    # 数据集名字
    parser.add_argument('-data-set-name', type=str, default=str(DATASRT_NAME),
                        help='Name of dataset.')

    # 路径配置
    parser.add_argument('-path', type=str, default='./data/' + str(DATASRT_NAME) + '/processed_data.csv',
                        help='Path to store data.')
    parser.add_argument('-split-path', type=str, default='./data/' + str(DATASRT_NAME) + '/split_data/',
                        help='The folder in which data is stored after partition. '
                             'Select in ["./data/[dataset_name]/split_data/", ].')

    # 数据划分
    parser.add_argument('-divided-flag', type=bool, default=False,
                        help='Whether the data is divided?')
    parser.add_argument('-divided-type', type=str, default='StratifiedKFold',
                        help='Divide the type of data. Select in [train_test_split, KFold, StratifiedKFold].'
                             'When select "train_test_split", k should is 0.')
    parser.add_argument('-k', type=int, default=2,
                        help='When divided type in [KFold, StratifiedKFold], select k?.')

    # 词向量训练
    parser.add_argument('-train_word2vec', type=bool, default=NEED_WORD2VEC,
                        help='Does the model need to train Wor2vec? [default: False].')
    if NEED_WORD2VEC:
        parser.add_argument('-cut-flag', type=bool, default=True,
                            help='Whether to carry out word segmentation? [default: False].')
        parser.add_argument('-cut-out-dir', type=str, default='./data/' + str(DATASRT_NAME) + '/Word2Vec/',
                            help='The text directory after word segmentation.')
        parser.add_argument('-word2vec-dir', type=str, default='./data/' + str(DATASRT_NAME) + '/Word2Vec/',
                            help='A directory that stores trained word vectors.')

    # 模型配置
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')  # 使用gpu？
    parser.add_argument('-static', type=bool, default=True,
                        help='whether to use static pre-trained word vectors [default: False]')  # 静态词向量，预训练词向量
    parser.add_argument('-non-static', type=bool, default=False,
                        help='whether to fine-tune static pre-trained word vectors')  # 预训练词向量微调
    parser.add_argument('-pretrained-path', type=str, default='./data/' + str(DATASRT_NAME) + '/Word2Vec/',
                        help='path of pre-trained word vectors')  # 预训练词向量路径：'D:\\word2vec\\'
    parser.add_argument('-pretrained-name', type=str, default="word2vec.txt",
                        help='filename of pre-trained word vectors')  # 预训练词向量文件名: # glove.6B.300d.txt word2vec.txt
    parser.add_argument('-multichannel', type=bool, default=True,
                        help='whether to use 2 channel of word vectors')  # 预训练模型的多通道参数。一个预训练，一个随机训练。
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-class-num', type=int, default=2, help='Class number [default: 2]')

    # 模型参数
    parser.add_argument('-model-name', type=str, default=MODEL_NAME,
                        help='name of the training model')
    parser.add_argument('-embedding-dim', type=int, default=200,
                        help='number of embedding dimension [default: 128]')  # 预训练词向量维度
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')
    # "Model parameters"
    if MODEL_NAME in ["TextCNN1d", "TextCNN"]:
        parser.add_argument('-filter-sizes', type=str, default='1,2,3',  # '3,4,5'
                            help='comma-separated filter sizes to use for convolution')
        parser.add_argument('-filter-num', type=int, default=200,
                            help='number of each size of filter [default 200]')
    elif MODEL_NAME in ["MLP", "FastText"]:
        parser.add_argument('-hidden-dim', type=int, default=16,
                            help='MLP-hidden-dim [default: 64]')
    elif MODEL_NAME in ["RNN", "RNNAttention", "RCNN", "TextRNN"]:
        parser = rnn_config(parser)
    elif MODEL_NAME == "CNNRNN":
        parser = rnn_config(parser)
        parser.add_argument('-multi-channel-num', type=int, default=5,
                            help='The number of channel.  [default: 5]')
    elif MODEL_NAME == "Transformer":
        pass
    elif MODEL_NAME == "BERT":
        parser.add_argument('-bert-path', type=str, default='D:\\BERT_MODEL\\ENGLISH\\bert-base-uncased\\',
                            help='Loading path of BERT pretraining model, including [config.json, pytorch_model.bin, '
                                 ', vocab.txt].')
        parser.add_argument('-bert-input-size', type=int, default=768,
                            help='In file "config.json" is "hidden_size".')
    elif MODEL_NAME == "BERT_ERNIE":
        parser.add_argument('-bert-path', type=str, default='D:\\BERT_MODEL\\ENGLISH\\ernie-2.0-en\\',
                            help='Loading path of ERNIE pretraining model, including [config.json, pytorch_model.bin, '
                                 ', vocab.txt].')
        parser.add_argument('-bert-input-size', type=int, default=768,
                            help='In file "config.json" is "hidden_size".')
    elif MODEL_NAME in ["BERTCNN", "BERTDPCNN"]:
        parser = bert_cnn_config(parser)
    elif MODEL_NAME in ['BertCNNEmb', 'BERTDPCNNEmb', "BertCNNEmbSim"]:
        parser = bert_cnn_config(parser)
        parser.add_argument('-rel-embedding-dim', type=int, default=128,
                            help='The number of text relationship embedding dim.  [default: 128]')
        # rnn_config(args=parser)
    parser.add_argument('-max-norm', type=float, default=1e-2, help='l2 constraint of parameters [default: 3.0]')

    # 训练参数
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 32].')  # 128 64
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')  # 0.0001
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=16,
                        help='how many steps to wait before testing [default: 100]')  # !!!!!! 调优发现32是最好的，batch_size的一半
    parser.add_argument('-early-stopping', type=int, default=2000,
                        help='iteration numbers to stop without performance increasing [default: 1000]')  # 就是在增加1000次迭代批次后性能没有改善，就停止
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

    # 评估参数
    parser.add_argument('-confusion-matrix', type=bool, default=True,
                        help='Whether to draw the confusion matrix of the best model')
    parser.add_argument('-show-score', type=bool, default=True,
                        help='Whether show score of the best model, confusion-matrix should is True.')
    parser.add_argument('-show-report', type=bool, default=True,
                        help='Whether show report of the best model, confusion-matrix should is True.')
    parser.add_argument('-average-type', type=str, default='macro',
                        help='The way the model is evaluated. select in ["binary", "macro", "weighted"]')
    return parser


def bert_cnn_config(args):
    args.add_argument('-bert-path', type=str, default='D:\\BERT_MODEL\\ENGLISH\\bert-base-uncased\\',  # bert-large-uncased  bert-base-uncased
                        help='Loading path of BERT pretraining model, including [config.json, pytorch_model.bin, '
                             ', vocab.txt].')
    args.add_argument('-bert-input-size', type=int, default=768,  # 1024  768
                        help='In file "config.json" is "hidden_size".')
    args.add_argument('-filter-sizes', type=str, default='1,2',   # '1,2,3'
                        help='comma-separated filter sizes to use for convolution')
    args.add_argument('-filter-num', type=int, default=512,   # 200
                        help='number of each size of filter [default 200]')
    args.add_argument('-bert-path2', type=str, default='D:\\BERT_MODEL\\ENGLISH\\bert-large-uncased\\',
                      help='Loading path of BERT pretraining model, including [config.json, pytorch_model.bin, '
                           ', vocab.txt].')
    args.add_argument('-bert-input-size2', type=int, default=1024,  # 1024  768
                      help='In file "config.json" is "hidden_size".')
    return args


def rnn_config(args):
    args.add_argument('-rnn-type', type=str, default='LSTM',
                      help='The selected RNN type, select in [LSTM GRU RNN]')
    args.add_argument('-bidirectional', type=bool, default=True,
                      help='Whether it is a bidirectional RNN [default: True]')
    args.add_argument('-num-layers', type=int, default=2,
                      help='RNN-num-layers [default: 2]')
    args.add_argument('-hidden-dim', type=int, default=256,
                      help='RNN-hidden-dim [default: 64]')
    return args


def main_conf():
    # 参数配置
    parser = argparse.ArgumentParser(description='To configure some parameters.')

    # 配置参数
    parser = config(parser)

    # 命名空间结束
    args = parser.parse_args()
    return args


# args = main_conf()
# for attr, value in sorted(args.__dict__.items()):
#     if attr in {'vectors'}:
#         continue
#     print('\t{}={}'.format(attr.upper(), value))

