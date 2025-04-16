import torch
import torch.nn as nn
import torch.nn.functional as F


class Line(nn.Module):
    def __init__(self, size, embed_dim=128, order=1):
        super(Line, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")

        self.embed_dim = embed_dim
        self.order = order
        self.nodes_embeddings = nn.Embedding(size, embed_dim)

        if order == 2:
            self.contextnodes_embeddings = nn.Embedding(size, embed_dim)
            # Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(
                -.5, .5) / embed_dim

        # Initialization
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(
            -.5, .5) / embed_dim

    def forward(self, v_i, v_j, negsamples, device):
        # (batch_size, ) ->  (batch_size, embedding_dim)
        v_i = self.nodes_embeddings(v_i).to(device)  # 负采样样本

        if self.order == 2:
            # (batch_size, ) ->  (batch_size, embedding_dim)
            v_j = self.contextnodes_embeddings(v_j).to(device)
            # (batch_size, negsamples.shape[1]) ->  (batch_size, negsamples.shape[1], embedding_dim)
            negativenodes = -self.contextnodes_embeddings(negsamples).to(device)

        else:
            # (batch_size, ) ->  (batch_size, embedding_dim)
            v_j = self.nodes_embeddings(v_j).to(device)
            # (batch_size, negsamples.shape[1]) ->  (batch_size, negsamples.shape[1], embedding_dim)
            negativenodes = -self.nodes_embeddings(negsamples).to(device)

        # 元素点乘: (batch_size, embedding_dim) * .. ->  ..
        mulpositivebatch = torch.mul(v_i, v_j)
        # (batch_size, embedding_dim) ->  (batch_size, ) -> (batch_size, )
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

        # (batch_size, 1, embedding_dim) * (batch_size, negativenodes.shape[1], embedding_dim)
        # -> (batch_size, negativenodes.shape[1], embedding_dim)
        mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
        # (batch_size, negativenodes.shape[1], embedding_dim) -> (batch_size, )
        negativebatch = torch.sum(
            F.logsigmoid(
                torch.sum(mulnegativebatch, dim=2)
            ),
            dim=1)
        # (batch_size, ) * .. -> ..
        loss = positivebatch + negativebatch
        return -torch.mean(loss)
