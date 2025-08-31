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
    """将密集张量 x 转换为稀疏格式"""
    x_typename = torch.typename(x).split('.')[-1]
    # getattr()用于返回一个对象属性值,x的typename
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def Comp_degree(A):
    """ compute degree matrix of a graph D """
    out_degree = torch.sum(A, dim=0)
    in_degree = torch.sum(A, dim=1)

    diag = torch.eye(A.size()[0]).cuda()

    degree_matrix = diag * in_degree + diag * out_degree - torch.diagflat(torch.diagonal(A))

    return degree_matrix


class GraphConv_Ortega(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128, dropout=0.5, alpha=0.2, concat=True,sub_sample=True, bn_layer=True):
        super(GraphConv_Ortega, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.non_local = NONLocalBlock1D(120, hidden_dim,sub_sample,bn_layer)
        self.GAT = GraphAttentionLayer(in_dim, out_dim,
                                       dropout,
                                       alpha,
                                       concat)
        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)

        #### Adding MLP to GCN
        # self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)
        # self.batchnorm = nn.BatchNorm1d(out_dim)

        # for i in range(num_layers):
        #     init.xavier_uniform_(self.MLP.linears[i].weight)
        #     init.constant_(self.MLP.linears[i].bias, 0)

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
       # features = self.GAT(features, A)
        if (len(A.shape) == 2):
            # A_norm = A + torch.eye(n).cuda()
            # torch.eye(n),为了生成n*n对角线全1矩阵，其余部分全0的二维数组
            A_norm = A
            deg_mat = Comp_degree(A_norm) # 求A的度矩阵D
            #detach() 返回一个new Tensor，只不过不再有梯度。cpu()将变量放在cpu上，仍为tensor
            #fractional_matrix_power()求矩阵的分数幂，deg_mat的-1/2次幂
            frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(),
                                                                    -0.5)).cuda()
            Laplacian = deg_mat - A_norm
            #matmul()矩阵乘积   L的归一化
            Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))
            # torch.eig()计算实方阵的特征值和特征向量。U表示特征向量
            landa, U = torch.linalg.eig(Laplacian_norm)
            # repeat(b,1,1)将张量U.t()在深度的方向重复b次 行和列的方向不变
            #这里t()
            repeated_U_t = U.t().repeat(b, 1, 1)
            repeated_U = U.repeat(b, 1, 1)
        else:
            repeated_U_t = []
            repeated_U = []
            for i in range(A.shape[0]):
                # A_norm = A[i] + torch.eye(n).cuda()
                A_norm = A[i]
                deg_mat = Comp_degree(A_norm)
                frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(),
                                                                        -0.5)).cuda()
                Laplacian = deg_mat - A_norm
                Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))

                landa, U = torch.linalg.eig(Laplacian_norm)

                repeated_U_t.append(U.t().view(1, U.shape[0], U.shape[1]))
                repeated_U.append(U.view(1, U.shape[0], U.shape[1]))
                # torch.cat()在给定维度上对输入的张量序列seq 进行连接操作。
            repeated_U_t = torch.cat(repeated_U_t)
            repeated_U = torch.cat(repeated_U)
            # torch.bmm()计算两个tensor的矩阵乘法，
        repeated_U_t=repeated_U_t.float()
        repeated_U = repeated_U.float()
        features=features.float()
        agg_feats = torch.bmm(repeated_U_t, features)
        #print(agg_feats.size())
        #### Adding MLP to GCN
        out = self.MLP(agg_feats.view(-1, d)).view(b, -1, self.out_dim)
        # out = self.GAT(agg_feats.view(-1,d), A)
        #out = self.GAT(agg_feats, A)
        out = self.non_local(out)
        out = torch.bmm(repeated_U, out)+out
        #out = self.batchnorm(out).view(b, -1, self.out_dim)

        return out


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

        ###Adj matrix
        self.Adj = adj

        ###List of GCN layers
        self.GCNs = torch.nn.ModuleList()
        self.GCNs.append(GraphConv_Ortega(self.input_dim, self.hidden_dim))
        for i in range(self.num_layers - 1):
            self.GCNs.append(GraphConv_Ortega(self.hidden_dim, self.hidden_dim))

        # Linear functions that maps the hidden representations to labels
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.Dropout(p=self.final_dropout),
            nn.PReLU(128),
            nn.Linear(128, 64),
            nn.Linear(64, output_dim))

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features.view(1, -1, self.input_dim) for graph in batch_graph], 0).to(
            self.device)
        A = F.relu(self.Adj)
        # Pa=nn.Parameter(A)
        # nn.init.constant_(Pa, 1e-6)
        # A=Pa+A
        h = X_concat
        for layer in self.GCNs:
            h = F.relu(layer(h, A))

        if (self.graph_pooling_type == 'mean'):
            pooled = torch.mean(h, dim=1)
        if (self.graph_pooling_type == 'max'):
            pooled = torch.max(h, dim=1)[0]
        if (self.graph_pooling_type == 'sum'):
            pooled = torch.sum(h, dim=1)

        score = self.classifier(pooled)

        return score


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh =torch.einsum('nvc,cx->nvx', h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # e = Wh1 + Wh2.T
        Wh2 =Wh2.permute(0,2,1).contiguous()
        e =Wh1+Wh2

        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):

        # :param x: (b, n, d)
        # :return:


        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
