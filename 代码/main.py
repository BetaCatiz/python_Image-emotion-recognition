import argparse
import torch
import torchtext.data as data
from torchtext.vocab import Vectors
import sys
import pandas as pd
import dataset
import model
import train
from config import main_conf
from vectors import train_vec
from model_berts import TrainBert
from model_self import TrainBertCNNEmb


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
args = main_conf()


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)  # 这个好像有缓存，在第一次加载过后第二次就特别快，牛啊
    return vectors


def load_dataset(args, text_field, label_field, **kwargs):
    train_dataset, dev_dataset = dataset.get_dataset(path=args.path, divided_flag=args.divided_flag, text_field=text_field,
                                                     label_field=label_field, split_dir=args.split_path,
                                                     divided_type=args.divided_type, k=args.k)
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)  # 加载词向量
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors, max_size=None, min_freq=1)  # 这个不知道在哪可以一步一步查找, 如果传**kwargs，可以看谁用了**kwargs，找到调用类的初始化方式
    else:
        text_field.build_vocab(train_dataset, dev_dataset)  # 生成词的计数表示
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter = data.Iterator.splits(
        (train_dataset, dev_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter, dev_iter


if __name__ == '__main__':
    args.cuda = args.device != -1 and torch.cuda.is_available()  # 是否使用cuda
    if not args.divided_flag:
        all_data = pd.read_csv(args.path, encoding='utf-8', header=0)
        dataset.divided_data(all_data=all_data, split_path=args.split_path, divided_type=args.divided_type, k=args.k)

    # TODO 1. bert 类模型的训练
    if args.model_name[:4].lower() == 'bert':
        print('Parameters:')
        for attr, value in sorted(args.__dict__.items()):
            if attr in {'vectors'}:
                continue
            print('\t{}={}'.format(attr.upper(), value))
        if 'cnn' in args.model_name.lower():
            args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]  # 使用的滤波器大小
        if args.model_name in ['BERT', "BERT_ERNIE"]:  # 训练BERT
            if args.model_name == 'BERT':
                model = model.BERT(args)
            elif args.model_name == 'BERT_ERNIE':
                model = model.ERNIE(args)
            print('Printing model...')
            print(model)
            bert_trainer = TrainBert(data_set_name=args.data_set_name, k=args.k)
            try:
                bert_trainer.train_model(args, model)
            except KeyboardInterrupt:
                print('Exiting from training early')
            sys.exit()
        elif args.model_name in ['BERTCNN', 'BERTDPCNN']:
            if args.model_name == 'BERTCNN':
                model = model.BERTCNN(args)
            elif args.model_name == 'BERTDPCNN':
                model = model.BERTDPCNN(args)
            print('Printing model...')
            print(model)
            bert_trainer = TrainBert(data_set_name=args.data_set_name, k=args.k)
            try:
                bert_trainer.train_model(args, model)
            except KeyboardInterrupt:
                print('Exiting from training early')
            sys.exit()
        elif args.model_name in ['BertCNNEmb', 'BERTDPCNNEmb', "BertCNNEmbSim"]:
            if args.model_name == 'BertCNNEmb':
                model = model.BertCNNEmb(args)
            elif args.model_name == 'BERTDPCNNEmb':
                model = model.BERTDPCNNEmb(args)
            elif args.model_name == 'BertCNNEmbSim':
                model = model.BertCNNEmbSim(args)
            print('Printing model...')
            print(model)
            bert_trainer = TrainBertCNNEmb(data_set_name=args.data_set_name, k=args.k)
            try:
                bert_trainer.train_model(args, model)
            except KeyboardInterrupt:
                print('Exiting from training early')
            sys.exit()

    # TODO 2. word2vec、glove类模型的训练
    if args.train_word2vec:
        print('Training Word2vec...')
        train_vec(cut_out_dir=args.cut_out_dir, word2vec_dir=args.word2vec_dir, cut_flag=args.cut_flag,
                  data_path=args.path)
    print('Loading data...')
    text_field = data.Field(lower=False, stop_words=stopwords)  # 添加停止词。
    label_field = data.Field(sequential=False)
    train_iter, dev_iter = load_dataset(args, text_field, label_field, device=-1, repeat=False, shuffle=True)
    args.vocabulary_size = len(text_field.vocab)  # 命名空间可以添加其它参数：词汇表大小  len(text_field.vocab)  不能减去1的
    if args.static:  # 使用静态词向量
        args.embedding_dim = text_field.vocab.vectors.size()[-1]  # 这边把词向量维度计算出来了
        args.vectors = text_field.vocab.vectors
    if args.multichannel:
        args.static = True
        args.non_static = True
    args.class_num = len(label_field.vocab) - 1  # 情感分类数
    if args.model_name in ["TextCNN1d", "TextCNN"]:
        args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]  # 使用的滤波器大小
    print('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        if attr in {'vectors'}:
            continue
        print('\t{}={}'.format(attr.upper(), value))

    print('Loading model...')
    if args.model_name == 'TextCNN1d':
        model = model.TextCNN1d(args)
    elif args.model_name == 'TextCNN':
        model = model.TextCNN(args)
    elif args.model_name == 'MLP':
        model = model.MLP(args)
    elif args.model_name == 'RNN':
        model = model.RNN(args)
    elif args.model_name == 'RNNAttention':
        model = model.RNNAttention(args)
    # elif args.model_name == 'RCNN':
    #     model = model.RCNN(args)
    elif args.model_name == 'FastText':
        model = model.FastText(args)
    elif args.model_name == 'CNNRNN':
        model = model.CNNRNN(args)
    elif args.model_name == 'Transformer':
        model = model.Transformer(args)
    elif args.model_name == 'TextRNN':
        model = model.TextRNN(args)
    print('Printing model...')
    print(model)

    if args.snapshot:
        print('\nLoading model from {}...\n'.format(args.snapshot))
        model.load_state_dict(torch.load(args.snapshot))
    if args.cuda:
        torch.cuda.set_device(args.device)
        text_cnn = model.cuda()
    try:
        train.train(train_iter, dev_iter, model, args)
    except KeyboardInterrupt:
        print('Exiting from training early')





