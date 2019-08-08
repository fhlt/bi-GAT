import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # features用稀疏矩阵的形式存储
    # idx_features_labels的第一列编号，最后一列为标签
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 标签转化为one-hot编码
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 第二个编号的论文引用第一个编号的论文，第一个编号的论文被引用
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 取第一列，即论文编号
    idx_map = {j: i for i, j in enumerate(idx)}  # 为论文重新编号
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 读取引文信息
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 构建邻接矩阵，存在引用关系的位置为1，稀疏矩阵的表示形式
    adj_in = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj_out = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 1], edges[:, 0])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix 构建一个对称的邻接矩阵
    # 有向图邻接矩阵不是对称阵
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)  # 特征进行归一化处理
    adj_in = normalize_adj(adj_in + sp.eye(adj_in.shape[0]))  # 并加上自环，邻接矩阵进行归一化处理
    adj_out = normalize_adj(adj_out + sp.eye(adj_out.shape[0]))  # 并加上自环，邻接矩阵进行归一化处理

    idx_train = range(140)  # 分配训练集，验证集、测试集
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj_in = torch.FloatTensor(np.array(adj_in.todense()))
    adj_out = torch.FloatTensor(np.array(adj_out.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_in, adj_out, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

