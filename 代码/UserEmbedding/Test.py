import argparse
from utils.utils import *
from utils.line import Line
from tqdm import trange
import torch
import torch.optim as optim
import sys
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-graph-path", type=str, default='./data/self_data/OMD/sentiment_consistency_edge.csv')  # "--graph_path", './data/erdosrenyi.edgelist'
    parser.add_argument("-save-path",  type=str, default='./model.pt')  # "--save_path",
    parser.add_argument("-lossdata-path", type=str, default='./loss.pkl')  # "--lossdata_path",

    # Hyperparams.
    parser.add_argument("-order", type=int, default=2)  # "--order",
    parser.add_argument("-negsamplesize", type=int, default=5)  # "--negsamplesize",
    parser.add_argument("-dimension", type=int, default=128)  # "--dimension",
    parser.add_argument("-batchsize", type=int, default=5)  # "--batchsize", 少的5，多的16
    parser.add_argument("-epochs", type=int, default=100)  # "--epochs", 少的100，多的5
    parser.add_argument("-learning-rate", type=float,  # "--learning_rate",
                        default=0.0001)  # As starting value in paper，default=0.025
    parser.add_argument("-negativepower", type=float, default=0.75)  # , "--negativepower"
    args = parser.parse_args()

    # Create dict of distribution when opening file
    edgedistdict, nodedistdict, weights, nodedegrees, maxindex = makeDist(
        args.graph_path, args.negativepower)

    edgesaliassampler = VoseAlias(edgedistdict)
    nodesaliassampler = VoseAlias(nodedistdict)

    batchrange = int(len(edgedistdict) / args.batchsize)
    print(maxindex)
    line = Line(maxindex + 1, embed_dim=args.dimension, order=args.order)

    # opt = optim.SGD(line.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    opt = optim.Adam(line.parameters(), lr=args.learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lossdata = {"it": [], "loss": []}
    it = 0
    embedding_data = {}
    print("\nTraining on {}...\n".format(device))
    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch))
        for b in trange(batchrange):
            samplededges = edgesaliassampler.sample_n(args.batchsize)
            batch = list(makeData(samplededges, args.negsamplesize, weights, nodedegrees,
                                  nodesaliassampler))
            batch = torch.LongTensor(batch)
            v_i = batch[:, 0]
            v_j = batch[:, 1]
            negsamples = batch[:, 2:]
            
            line.zero_grad()
            loss = line(v_i, v_j, negsamples, device)
            print(loss)

            # if it % 100 == 0:
            #     embedding = line.nodes_embeddings(v_i).cpu().detach().numpy()
            #
            #     figure = plt.figure()
            #     plt.scatter(embedding[:, 0], embedding[:, 1], s=14)
            #     plt.show()
            #     print(loss)

            loss.backward()
            opt.step()

            lossdata["loss"].append(loss.item())
            lossdata["it"].append(it)
            it += 1

    print("\nDone training, saving model to {}".format(args.save_path))
    torch.save(line.state_dict(), "{}".format(args.save_path))


    print("Saving loss data at {}".format(args.lossdata_path))
    with open(args.lossdata_path, "wb") as ldata:
        pickle.dump(lossdata, ldata)
    sys.exit()
