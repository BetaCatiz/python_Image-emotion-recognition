import pandas as pd


def map_data(path, out_dict, out_path):
    '''
        改变index下标并存储映射关系；将下标从小到大有序排列后，转换为0-max的下标索引；
    :param path: 需要进行映射从数据path
    out_dict: 存储映射后的dict
    out_path: 映射后的数据
    :return:

    '''
    data = pd.read_csv(path, header=0, encoding='utf-8')
    ori_data = data['ori'].values.tolist()
    dir_data = data['dir'].values.tolist()
    all_p = sorted(list(set(ori_data + dir_data)))
    print(all_p)
    dict_data = dict([(i, j) for i, j in zip(all_p, list(range(len(all_p))))])
    print(dict_data)
    print(dict_data[128])


path = './data/self_data/b.csv'
map_data(path)


