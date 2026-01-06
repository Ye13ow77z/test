import numpy as np
import scipy.sparse as sp
import torch
import os

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy.sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def load_citeseer(path='./DCRN/dataset/citeseer/', device='cpu'):
    print(f"Loading Citeseer dataset from {path}...")
    
    # 1. 读取 Content 文件 (ID, Features, Label)
    content_file = os.path.join(path, 'citeseer.content')
    if not os.path.exists(content_file):
        raise FileNotFoundError(f"Could not find {content_file}")

    print("Parsing content file...")
    idx_features_labels = np.genfromtxt(content_file, dtype=np.dtype(str))
    
    # 提取特征 (跳过第一列ID和最后一列Label)
    features_raw = idx_features_labels[:, 1:-1].astype(np.float32) 
    features = sp.csr_matrix(features_raw, dtype=np.float32)
    
    # 提取标签 (最后一列)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # 构建 ID 到 索引 的映射
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    
    # 2. 读取 Cites 文件 (构建图)
    cites_file = os.path.join(path, 'citeseer.cites')
    print("Parsing cites file...")
    edges_unordered = np.genfromtxt(cites_file, dtype=np.dtype(str))
    
    # === 关键步骤：过滤孤立节点 ===
    # 只保留那些 Source 和 Target 都在 idx_map 中的边
    edges = []
    for edge in edges_unordered:
        if edge[0] in idx_map and edge[1] in idx_map:
            edges.append([idx_map[edge[0]], idx_map[edge[1]]])
    
    edges = np.array(edges, dtype=np.int32)
    
    # 构建邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # 3. 后处理 (对称化 + 归一化)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # 添加自环
    adj = adj + sp.eye(adj.shape[0])
    
    # 归一化特征
    features = normalize(features)
    
    # 转换为 Tensor
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)
    
    # 归一化邻接矩阵并转稀疏 Tensor
    adj_norm = normalize_adj(adj)
    adj_sparse = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
    
    # 构造用于重构 Loss 的 adj_label (Dense)
    adj_label = torch.FloatTensor(np.array(adj.todense())).to(device)

    return adj_sparse, features, labels, adj_label

if __name__ == "__main__":
    # 测试代码
    try:
        adj, feat, label, adj_l = load_citeseer()
        print("Success!")
        print(f"Nodes: {feat.shape[0]}")
        print(f"Features: {feat.shape[1]}")
        print(f"Classes: {label.max().item() + 1}")
    except Exception as e:
        print(e)