import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W_in = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_in.data, gain=1.414)
        self.W_out = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_out.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    # 接下来只需要修改这里就OK
    def forward(self, input, adj_in, adj_out):
        # torch.mm是两个矩阵相乘，torch.mul是两个矩阵对应元素相乘（矩阵必须大小相同）
        # 通过一维卷积增加非线性，赋予不同权重
        h_in = torch.mm(input, self.W_in)
        h_out = torch.mm(input, self.W_out)
        N = h_in.size()[0]  # N为节点数量
        # repeat在第1个维重复N次
        # view的作用是reshape
        # 在第一个维度上cat
        # 将a_input变成一个形状为（N，N，2*features)，表示N个节点一一对应的特征concat结果
        a_input_in = torch.cat([h_in.repeat(1, N).view(N * N, -1), h_in.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        a_input_out = torch.cat([h_out.repeat(1, N).view(N * N, -1), h_out.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # 加权之后，将第二个维度压缩掉，e为N*N的矩阵，表示两两节点之间attention的结果
        e_in = self.leakyrelu(torch.matmul(a_input_in, self.a).squeeze(2))
        e_out = self.leakyrelu(torch.matmul(a_input_out, self.a).squeeze(2))
        
        zero_vec = -9e15*torch.ones_like(e_in)
        # torch.where(判断条件，满足条件的设置值，不满足条件的设置值)
        # 如果邻接，则置为e，否则置为无穷小，为什么不直接置为0？
        attention_in = torch.where(adj_in > 0, e_in, zero_vec)
        attention_out = torch.where(adj_out > 0, e_out, zero_vec)

        attention_in = F.softmax(attention_in, dim=1)
        attention_out = F.softmax(attention_out, dim=1)
        attention_in = F.dropout(attention_in, self.dropout, training=self.training)
        attention_out = F.dropout(attention_out, self.dropout, training=self.training)
        h_prime_in = torch.matmul(attention_in, h_in)
        h_prime_out = torch.matmul(attention_out, h_out)
        # 聚合
        h_prime = (h_prime_in + h_prime_out) / 2

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
