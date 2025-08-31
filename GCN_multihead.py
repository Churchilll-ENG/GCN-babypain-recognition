import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import eig
import sys
from torch.nn import init
from scipy.linalg import fractional_matrix_power

sys.path.append("models/")
from mlp import MLP
from properties.choice_A import choice_A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def Comp_degree(A):
    """ compute degree matrix of a graph """
    out_degree = torch.sum(A, dim=0)
    in_degree = torch.sum(A, dim=1)

    diag = torch.eye(A.size()[0]).cuda()

    degree_matrix = diag * in_degree + diag * out_degree - torch.diagflat(torch.diagonal(A))

    return degree_matrix


def feature_U_compute(A):
    # 把计算A的特征向量拿出来只计算一次，然后这个只适应A不变的情况，
    A_norm = A
    deg_mat = Comp_degree(A_norm)
    frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(),
                                                            -0.5)).cuda()
    frac_degree = frac_degree.float()

    Laplacian = deg_mat - A_norm
    Laplacian = Laplacian.float()

    Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))

    landa, U = torch.linalg.eig(Laplacian_norm)

    return U

class GraphConv_Ortega(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        super(GraphConv_Ortega, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)

        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)

        #### Adding MLP to GCN
        # self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)
        # self.batchnorm = nn.BatchNorm1d(out_dim)

        # for i in range(num_layers):
        #     init.xavier_uniform_(self.MLP.linears[i].weight)
        #     init.constant_(self.MLP.linears[i].bias, 0)

    def forward(self, features, U):
        b, n, d = features.shape
        assert (d == self.in_dim)

        repeated_U_t = U.t().repeat(b, 1, 1)
        repeated_U = U.repeat(b, 1, 1)

        repeated_U_t = repeated_U_t.float()
        repeated_U = repeated_U.float()
        features = features.float()
        agg_feats = torch.bmm(repeated_U_t, features)

        #### Adding MLP to GCN
        out = self.MLP(agg_feats.view(-1, d)).view(b, -1, self.out_dim)
        out = torch.bmm(repeated_U, out)
        # out = self.batchnorm(out).view(b, -1, self.out_dim)

        return out


class GraphConv_multi(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        super(GraphConv_multi, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.MLP1 = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.MLP2 = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.MLP3 = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.init_weights(self.MLP1)
        self.init_weights(self.MLP2)
        self.init_weights(self.MLP3)

        # 初始化权重
        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.weight3 = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(p=0.5)
    def init_weights(self, mlp):
        for i in range(len(mlp.linears)):
            init.xavier_uniform_(mlp.linears[i].weight)
            init.constant_(mlp.linears[i].bias, 0)
    def get_one_mlpout(self, mlp, features,U):
        b, n, d = features.shape
        assert (d == self.in_dim)

        repeated_U_t = U.t().repeat(b, 1, 1)
        repeated_U = U.repeat(b, 1, 1)

        repeated_U_t = repeated_U_t.float()
        repeated_U = repeated_U.float()
        features = features.float()
        agg_feats = torch.bmm(repeated_U_t, features)
        out = mlp(agg_feats.view(-1, d)).view(b, -1, self.out_dim)

        out = torch.bmm(repeated_U, out)
        return out
    def forward(self, features, U1,U2,U3):
        out1 = self.get_one_mlpout(self.MLP1, features, U1)
        out2 = self.get_one_mlpout(self.MLP2, features, U2)
        out3 = self.get_one_mlpout(self.MLP3, features, U3)

        # 计算加权求和
        weights = torch.softmax(torch.tensor([self.weight1, self.weight2, self.weight3]), dim=0)
        merged_out = weights[0] * out1 + weights[1] * out2 + weights[2] * out3

        # 恢复到 (b, n, output_dim)
        # out = merged_out.view(b, n, -1)
        return merged_out

class Graph_CNN_ortega(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, final_dropout,
                 graph_pooling_type, device, adj):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(Graph_CNN_ortega, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.Adj = adj.to(self.device)
        self.Adj2 = choice_A('gao', 3).to(self.device)
        self.Adj3 = choice_A('gao', 3).to(self.device)

        ###List of GCN layers
        self.GCNs = torch.nn.ModuleList()
        self.GCNs.append(GraphConv_multi(self.input_dim, self.hidden_dim))
        for i in range(self.num_layers - 1):
            self.GCNs.append(GraphConv_multi(self.hidden_dim, self.hidden_dim))

        # Linear functions that maps the hidden representations to labels
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.Dropout(p=self.final_dropout),
            nn.PReLU(128),
            nn.Linear(128, output_dim))
        
        self.U1 = feature_U_compute(self.Adj)
        self.U2 = feature_U_compute(self.Adj2)
        self.U3 = feature_U_compute(self.Adj3)

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features.view(1, -1, self.input_dim) for graph in batch_graph], 0).to(
            self.device)
        A = F.relu(self.Adj)
        Pa = nn.Parameter(A)

        A = Pa + A
        h = X_concat
        for layer in self.GCNs:
            h = F.relu(layer(h, self.U1,self.U2,self.U3))

        if (self.graph_pooling_type == 'mean'):
            pooled = torch.mean(h, dim=1)
        if (self.graph_pooling_type == 'max'):
            pooled = torch.max(h, dim=1)[0]
        if (self.graph_pooling_type == 'sum'):
            pooled = torch.sum(h, dim=1)

        score = self.classifier(pooled)

        return score
