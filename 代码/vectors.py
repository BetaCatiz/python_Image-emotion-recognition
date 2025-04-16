import gensim
import pandas as pd
import jieba
import logging
import re
from gensim.models import Word2Vec

# import nltk
# nltk.word_tokenize()
# from nltk.corpus import stopwords
# stops = stopwords.words('english')
# from nltk.stem import LancasterStemmer
# stemmerlan = LancasterStemmer()  # 提取词干
# stemmerlan.stem('because')  # 'becaus'

jieba.setLogLevel(logging.INFO)
regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')  # 特殊符号
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
             'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
             "it's", 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
             'these', 'those', 'am',
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
             'doing', 'a', 'an', 'the',
             'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
             'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
             'off', 'over', 'under',
             'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
             'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
             's', 't', 'can', 'will',
             'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
             "aren't", 'couldn',
             "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
             'isn', "isn't", 'ma',
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
             'wasn', "wasn't", 'weren',
             "weren't", 'won', "won't", 'wouldn', "wouldn't"]
regex1 = re.compile(r'http : //[a-zA-Z0-9.?/&=:]*')  # url


def train_vec(word2vec_dir=None, cut_out_dir=None, cut_flag=False, **kwargs):
    if not cut_flag:
        get_cut_data(kwargs['data_path'], cut_out_dir)
    sentences = gensim.models.word2vec.LineSentence(cut_out_dir + 'cut_out.txt')
    model = Word2Vec(sentences, vector_size=200, sg=1, epochs=8, min_count=1)
    model.wv.save_word2vec_format(word2vec_dir + "word2vec.txt", binary=False)


def f(text):
    text = regex1.sub(' ', text)
    text = regex.sub(' ', text)

    word_list = [word.lower() for word in jieba.cut(text) if (word.strip()) and (word.lower() not in stopwords)]
    text = " ".join(word_list)
    return text


# def english_f(text):  # 有时候删除太多停止词、网址啥的好像也并不能改善模型
#     text = " ".join([i for i in text.split() if (i[:4] != 'http') and (i[:1] != '#') and (i[:1] != '@')])
#     text = regex.sub(' ', text)
#
#     word_list = [word.lower() for word in text.split() if (word.strip()) and (word.lower() not in stopwords)]
#     text = " ".join(word_list)
#     return text


def get_cut_data(path, out_dir):
    all_data = pd.read_csv(path, encoding='utf-8', header=0)
    cut_data = all_data['text']
    cut_data = cut_data.apply(f)
    cut_data.to_csv(out_dir + 'cut_out.txt', encoding='utf-8', mode='w', header=False, index=False)


# train_vec(cut_out_dir='./data/HCR/Word2Vec/', word2vec_dir='./data/HCR/Word2Vec/', cut_flag=False,
#           data_path='./data/HCR/processed_data.csv')
