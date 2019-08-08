import torch 
import torch.nn as nn
import torch.nn.functional as F 
from layers import GraphAttentionLayer

class GraphAttentionLayer(nn.Module): 
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        '''
        :param input: 输入特征 (batch,in_features)
        :param adj:  邻接矩阵 (batch,batch)
        :return: 输出特征 (batch,out_features)
        '''
        h = torch.mm(input, self.W) # (batch,out_features)
        N = h.size()[0] # batch
        # 将不同样本之间两两拼接，形成如下矩阵
        # [[结点1特征，结点1特征],
		#  [结点1特征，结点2特征],
		#   ......
		#  [结点1特征，结点j特征],
		#  [结点2特征，结点1特征],
		#  [结点2特征，结点2特征],
		#   ......
		#   ]
        a_input = torch.cat([h.repeat(1, N)   # (batch,out_features*batch)
                            .view(N * N, -1), # (batch*batch,out_features)
                             h.repeat(N, 1)], # (batch*batch,out_features)
                            dim=1).view(N, -1, 2 * self.out_features) # (batch,batch,2 * out_features)
        # 通过刚刚的矩阵与权重矩阵a相乘计算每两个样本之间的相关性权重，最后再根据邻接矩阵置零没有连接的权重
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (batch,batch)
        # 置零的mask
        zero_vec = -9e15*torch.ones_like(e) # (batch,batch)
        attention = torch.where(adj > 0, e, zero_vec) # (batch,batch) 有相邻就为e位置的值，不相邻则为0
        attention = F.softmax(attention, dim=1)  # (batch,batch)
        attention = F.dropout(attention, self.dropout, training=self.training) # (batch,batch)
        h_prime = torch.matmul(attention, h) # (batch,out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
