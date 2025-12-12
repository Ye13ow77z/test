import numpy as np
import torch
import scipy.sparse as sp
import os
from sklearn.decomposition import PCA

def get_adj_normalized(adj):
    """
    [通用归一化] 计算 GCN 标准对称归一化矩阵: D^-0.5 * (A+I) * D^-0.5
    相比原来的 D^-1 * (A+I)，这个对 GAE/GCN 训练更稳定
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0]) # A + I
    rowsum = np.array(adj_.sum(1))
    
    # 防止除以0
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # 如果有无穷大(孤立点)，替换为0
    degree_mat_inv_sqrt.data[np.isinf(degree_mat_inv_sqrt.data)] = 0.
    
    # D^-0.5 * (A+I) * D^-0.5
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    
    return torch.FloatTensor(adj_normalized.toarray())

def load_graph_data(dataset_name, path='./DCRN/dataset/', use_pca=True, pca_dim=50, device='cpu'):
    """
    通用数据加载函数
    Args:
        dataset_name: 数据集名称 (e.g., 'cora', 'dblp', 'acm')
        path: 数据集根目录
        use_pca: 是否使用 PCA 降维 (Cora 推荐 False, 大图推荐 True)
        pca_dim: 降维维度
        device: 加载到的设备
    Returns:
        adj_norm: 归一化后的邻接矩阵 (用于 GCN 输入)
        feat: 节点特征
        label: 节点标签
        adj: 原始邻接矩阵 (用于重构 Loss)
    """
    # 1. 构造路径
    root_dir = os.path.join(path, dataset_name)
    print(f"Loading {dataset_name} from {root_dir}...")
    
    if not os.path.exists(root_dir):
        # 尝试去掉 dataset_name 文件夹这一层，直接在 path 下找 (兼容性处理)
        if os.path.exists(os.path.join(path, f"{dataset_name}_feat.npy")):
            root_dir = path
        else:
            raise FileNotFoundError(f"Directory not found: {root_dir}")

    # 2. 智能文件加载 (自动处理文件名带不带前缀的问题)
    def load_file(suffix):
        # 尝试1: dataset_feat.npy (如 cora_feat.npy)
        p1 = os.path.join(root_dir, f"{dataset_name}_{suffix}.npy")
        if os.path.exists(p1):
            return np.load(p1, allow_pickle=True)
        
        # 尝试2: feat.npy (通用命名)
        p2 = os.path.join(root_dir, f"{suffix}.npy")
        if os.path.exists(p2):
            return np.load(p2, allow_pickle=True)
            
        raise FileNotFoundError(f"Could not find {suffix} file in {root_dir}")

    feat = load_file('feat')
    label = load_file('label')
    adj = load_file('adj')

    # 3. 处理特征 (PCA)
    if use_pca:
        if feat.shape[1] > pca_dim:
            print(f"Applying PCA: {feat.shape[1]} -> {pca_dim}")
            pca = PCA(n_components=pca_dim)
            feat = pca.fit_transform(feat)
        else:
            print(f"PCA skipped: Feature dim {feat.shape[1]} <= {pca_dim}")
    
    # 类型转换
    if feat.dtype == np.object_: feat = feat.astype(np.float32)
    feat = torch.FloatTensor(feat).to(device)

    # 4. 处理标签
    # 有些数据集 label 是 [[1], [0]...] 这种格式，需要展平
    if label.ndim > 1:
        if label.shape[1] == 1:
            label = label.reshape(-1)
        # 如果是 One-Hot 编码 (N, C)，转为 Class Index (N,)
        elif label.shape[1] > 1:
            label = np.argmax(label, axis=1)

    label = torch.LongTensor(label).to(device)

    # 5. 处理邻接矩阵 (返回两种格式)
    # 格式 A: 原始 Adj (用于计算重构损失, 稀疏矩阵转换)
    if sp.issparse(adj):
        adj = adj.tocoo()
    else:
        adj = sp.coo_matrix(adj)
    
    # 转为 Tensor 用于重构 Loss 计算 (Pos Weight 计算需要用到)
    # 注意：如果图特别大，这里转 Dense 可能会爆显存，Cora 这种小图没问题
    adj_dense_tensor = torch.FloatTensor(adj.toarray()).to(device)

    # 格式 B: 归一化 Adj (用于 GCN 输入)
    print("Normalizing Adjacency Matrix...")
    adj_norm = get_adj_normalized(adj).to(device)
    
    print(f"Data Loaded: Nodes={feat.shape[0]}, Features={feat.shape[1]}, Classes={len(torch.unique(label))}")
    
    return adj_norm, feat, label, adj_dense_tensor