import torch
from utils.utils import VoseAlias, makeDist, negSampleBatch, makeData
from utils.line import Line
from tqdm import trange
import json


batchsize = 5  # 和前面一致
dimension = 128
order = 2
negativepower = 0.75
graph_path = './data/self_data/OMD/sentiment_consistency_edge.csv'  # './data/self_data/HCR/d.csv'
negsamplesize = 5


edgedistdict, nodedistdict, weights, nodedegrees, maxindex = makeDist(graph_path, negativepower)
edgesaliassampler = VoseAlias(edgedistdict)
nodesaliassampler = VoseAlias(nodedistdict)
batchrange = int(len(edgedistdict) / batchsize)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

line = Line(maxindex + 1, embed_dim=dimension, order=order)
lossdata = {"it": [], "loss": []}
it = 0

line.load_state_dict(torch.load('./model.pt'))
line.eval()
embedding_data = dict({})
for b in trange(batchrange):
    samplededges = edgesaliassampler.sample_n(batchsize)
    batch = list(makeData(samplededges, negsamplesize, weights, nodedegrees, nodesaliassampler))
    batch = torch.LongTensor(batch)
    v_i = batch[:, 0]
    v_j = batch[:, 1]
    negsamples = batch[:, 2:]
    loss = line(v_i, v_j, negsamples, device)
    embedding = line.nodes_embeddings(v_i).cpu().detach().numpy()
    embedding = embedding.tolist()
    for (node, emb) in zip(v_i.numpy().tolist(), embedding):
        embedding_data[node] = emb

embedding_data = sorted(embedding_data.items(), key=lambda x: x[0])
json_str = json.dumps(embedding_data, indent=4)
print(json_str)
with open('./embedding.json', 'w', encoding='utf-8') as fp:
    fp.write(json_str)
    fp.close()
