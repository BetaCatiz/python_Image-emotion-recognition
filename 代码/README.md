# 1. 文件说明
## 1.1 文件夹
- './data': 存储数据，里边建立你自己的数据集的名称
- './data/[dataset_name]/processed_data.csv：processed_data.csv 为你原始的数据集，把名称改为这个就行
- './data/[dataset_name]/split_data/'：划分后的数据（[dataset_name]代表你数据集的名称）
- './data/[dataset_name]/split_data/KFold/'：选取不考虑数据类别数量的随机分类方式（2/8）
- './data/[dataset_name]/split_data/StratifiedKFold/'：选取考虑数据类别数量的随机分类方式，尽量做到分的数据中各个类别数据的一致（2/8）
- './data/[dataset_name]/split_data/train_test_split/'：随机将数据划分的测试集和训练集（2/8）
- './data/[dataset_name]/split_data/KFold/1/'：选取不考虑数据类别数量的随机分类方式第1折数据，里边有’train.csv'和'test.csv'为你的训练集和测试集数据
- './data/[dataset_name]/relation_embedding_data/'：存储训练好的文本关系嵌入数据（原始嵌入字典（embdeeing.json）及对应与原始数据集的嵌入(processed_data.csv)）
- './model/'：目前没有存什么，自己可以存一些预训练模式啥的
- './picture/'：记录自己训练时候的训练及配置信息，自己每次把打印出来的信息复制保存到这里（我一般习惯命名为：时间.txt）
- './snapshot/'：用于保存训练中的最优模型
## 1.2 文件
- config.py：包括模型及训练的基本配置参数（argparse）
- dataset.py：对数据的预处理
- main.py：从这里开始运行
- model.py：一些常用的神经网络模型
- model_berts.py：由于bert和其余模型训练时候的输入方式的差异性，这里是bert系列模型的训练入口（在main.py中已经调用）
- train.py：训练模型
- vectors.py：训练模型的词向量

# 2. 配置说明
1. 数据集每一列的列名与dataset.py文件中get_dataset()函数中的下面代码对应：\
``fields=[
    ('username', None),
    ('text', text_field),
    ('label', label_field),
]
'text' 和 'label'是必须有的，对应与文本和情感标签（或者类别标签：0 1 2...）；
``
2. config.py中的参数配置说明：
- AVERAGE_TYPE=macro : 评估模式时候用的方式，参见sklearn.metrics.xxx
- BATCH_SIZE=64 : 训练的批次大小
- CONFUSION_MATRIX=True：是否显示混淆矩阵
- CUDA=False：是否使用GPU
- CUT_FLAG=False：在训练词向量时候，是否完成分词
- CUT_OUT_DIR=./data/OMD/Word2Vec/：在训练词向量时候，完成分词后存放的目录
- DEVICE=-1 : 使用GPU？，-1为使用cpu
- DIVIDED_FLAG=False：数据是否完成训练集和验证集的划分
- DIVIDED_TYPE=StratifiedKFold：数据是在划分训练集和验证集的划分时候，采取的方式（直接划分、不管类别数量的5折划分、均衡数量的5折划分）
- DROPOUT=0.5：模型使用的dropout率
- EARLY_STOPPING=1000：模型在经过1000步骤后在验证集上没有得到性能的提升就停止训练，返回best模型
- EMBEDDING_DIM=200：词嵌入的维度，如果是加载的词向量维度，这边也需要对应的修正
- EPOCHS=256：训练的epoch数
- FILTER_NUM=200：在使用卷积网络时候，使用的滤波器数量
- FILTER_SIZES=[1, 2, 3]：在使用卷积网络时候，使用的卷积核大小
- K=2：在对数据进行划分后（不管类别数量的5折划分、均衡数量的5折划分），选取第几折的数据进行模型训练与验证
- LOG_INTERVAL=1：相隔LOG_INTERVAL步后输出一次此时的训练集loss和acc
- LR=0.001：模型的学习率
- MAX_NORM=3.0：L2正则化使用的参数
- MODEL_NAME=TextCNN：使用的model的名称是什么
- MULTICHANNEL=True：是否使用多通道的词向量，这个一般一个通道是随机的词向量，一个是可以微调的词向量。
- NON_STATIC=True：是否对预训练的词向量进行微调
- PATH=./data/OMD/processed_data.csv：数据的存放目录，将自己的数据文件名字改到这里就行
- PRETRAINED_NAME=word2vec.txt：预训练词向量模型的模型名称
- PRETRAINED_PATH=./data/OMD/Word2Vec/：预训练词向量模型的模型路径
- SAVE_BEST=True：保存最好的模型
- SAVE_DIR=snapshot：保存最好模型的路径
- SHOW_REPORT=True：是否显示模型的report
- SHOW_SCORE=True：是否显示模型的score
- SNAPSHOT=None：snapshot的文件名
- SPLIT_PATH=./data/OMD/split_data/：划分数据存储的文件路径
- STATIC=True：是否使用静态词向量（即是否使用预训练的词向量模型）
- TEST_INTERVAL=100：TEST_INTERVAL步骤后测试一侧模型在验证集的情况
- TRAIN_WORD2VEC=True：是否需要训练词向量
- VOCABULARY_SIZE=4445：词汇表大小（程序自行给出）
- WORD2VEC_DIR=./data/OMD/Word2Vec/：训练的词向量存储的地址

# 3. 怎么生成社交嵌入向量：
- ./UserEmbedding/Test.py 是训练社交向量的主函数，运行完成后产生loos.pkl和model.pt为训练好的模型；
- 接下来就是加载模型，然后存储数据为：用户节点：向量表示的形式，存储在embedding.json；
- 将embedding.json放在./data/[数据集名称]/relation_embedding_data/下
- 在model_self.py中有生成每个微博嵌入关系的主程序，可以将其按照之前文本的划分规则划分为对应的数据，存储在./data/[数据集名称]/relation_embedding_data/下
