import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # 确保 adj 是稀疏矩阵 (coo_matrix)，方便后续处理
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)
    else:
        adj = adj.tocoo()
        
    rowsum = np.array(adj.sum(1))
    # 避免除以 0
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # D^-0.5 * A * D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy.sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    # === 修复 UserWarning: 使用 sparse_coo_tensor 替代 SparseTensor ===
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def load_acm(path='./DCRN/dataset/ACM3025.mat', device='cpu'):
    print(f"Loading ACM dataset from {path}...")
    try:
        data = sio.loadmat(path)
    except FileNotFoundError:
        print(f"Error: File not found at {path}. Please check the path.")
        sys.exit()

    # 1. 获取特征 (Features)
    if 'feature' in data:
        features = data['feature']
    elif 'features' in data:
        features = data['features']
    else:
        features = data['X']
        
    # === 修复1：健壮的类型转换 ===
    if sp.issparse(features):
        features = features.todense()
    # 确保转为 numpy array (处理 np.matrix 的情况)
    features = np.array(features) 
    features = torch.FloatTensor(features).to(device)

    # 2. 获取标签 (Labels)
    if 'label' in data:
        labels = data['label']
    elif 'gnd' in data:
        labels = data['gnd']
    elif 'Y' in data:
        labels = data['Y']
        
    # 如果是 one-hot，转为 1D
    if labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    else:
        labels = labels.reshape(-1)
        
    labels = torch.LongTensor(labels).to(device)

    # 3. 构建邻接矩阵 (Adjacency)
    # 使用 PAP (Paper-Author-Paper) 作为主要视图
    if 'PAP' in data:
        adj = data['PAP']
    elif 'A' in data:
        adj = data['A']
    else:
        print("Warning: No Adjacency Matrix found (PAP). Using Identity.")
        adj = sp.eye(features.shape[0])

    # === 修复2：处理邻接矩阵类型 ===
    # 如果已经是 dense (numpy.matrix 或 ndarray)，先转为 sparse 以便加自环和归一化
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)
    
    # 确保是对角线为1 (Self-loop)
    # data['PAP'] 读取出来可能不是 integer 类型，导致 .data 修改报错，统一转 float
    adj = adj.astype(np.float32) 
    
    # 加上自环 (adj + I)
    # 技巧：(adj > 0) 确保只有 0/1，防止多次叠加权重爆炸
    adj = adj + sp.eye(adj.shape[0])
    # 再次二值化 (可选，取决于 PAP 是否包含权重，通常设为 1 即可)
    # adj.data[adj.data > 0] = 1.0 
    
    # 归一化并转为 Sparse Tensor (用于 GCN 传播)
    adj_norm = normalize_adj(adj)
    adj_sparse = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)

    # 4. 构造用于重构 Loss 的 adj_label
    # 原始 adj (加了自环的) 转为 Dense Tensor
    if sp.issparse(adj):
        adj_dense_np = adj.todense()
    else:
        adj_dense_np = adj
        
    # 再次确保转为 np.array 
    adj_label = torch.FloatTensor(np.array(adj_dense_np)).to(device)

    return adj_sparse, features, labels, adj_label

if __name__ == "__main__":
    # 简单测试块
    path = './DCRN/dataset/ACM3025.mat'
    print(f"Testing loading from: {path}")
    try:
        adj, features, labels, adj_label = load_acm(path)
        print("Success!")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Adj Sparse indices: {adj._indices().shape}")
        print(f"Adj Label shape: {adj_label.shape}")
    except Exception as e:
        print(f"Failed again: {e}")