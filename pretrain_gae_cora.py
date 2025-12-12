import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import scipy.sparse as sp

# === 1. 定义工具函数：归一化邻接矩阵 ===
def normalize_adj(adj):
    """
    计算 GCN 标准归一化矩阵: D^-0.5 * (A+I) * D^-0.5
    """
    # 加上自环 A + I
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    
    # 计算度矩阵 D
    rowsum = np.array(adj_.sum(1))
    
    # 计算 D^-0.5
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    
    # 计算 D^-0.5 * (A+I) * D^-0.5
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    
    return torch.FloatTensor(adj_normalized.toarray())

# === 2. 定义 GCN 层 ===
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.mm(adj, x)
        return x

# === 3. 定义 GAE 模型 ===
class GAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_z):
        super(GAE, self).__init__()
        # 编码器: GCN两层
        self.gcn1 = GraphConvLayer(n_input, n_hidden)
        self.gcn2 = GraphConvLayer(n_hidden, n_z)

    def forward(self, x, adj):
        # Encoder
        hidden = F.relu(self.gcn1(x, adj))
        z = self.gcn2(hidden, adj)
        
        # Decoder (Inner Product)
        # 这里为了数值稳定性，我们不直接在模型里做 Sigmoid
        # 而是返回 Logits (z * z.t)，在 Loss 函数里做 Sigmoid
        adj_logits = torch.matmul(z, z.t())
        return adj_logits, z

# === 4. 数据加载 ===
def load_cora_data(path='./DCRN/dataset/cora', device='cpu'):
    print(f"Loading Cora from {path}...")
    
    # 确保文件名和你截图里的一致
    feat = np.load(os.path.join(path, 'cora_feat.npy'))
    adj = np.load(os.path.join(path, 'cora_adj.npy'))
    
    # 1. 制作归一化的 Adj (用于输入 GCN)
    adj_norm = normalize_adj(adj).to(device)
    
    # 2. 原始特征转 Tensor
    feat = torch.FloatTensor(feat).to(device)
    
    # 3. 原始 Adj 转 Tensor (作为 Label)
    adj_label = torch.FloatTensor(adj).to(device)
    
    return adj_norm, feat, adj_label

# === 5. 主训练逻辑 ===
def pretrain_gae(dataset_name='cora'):
    # 参数配置
    n_input = 1433
    n_hidden = 256
    n_z = 50
    lr = 0.001
    epochs = 200  # <--- 增加到 200 轮，50轮太短学不到东西
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    adj_norm, feat, adj_label = load_cora_data(path='./DCRN/dataset/cora', device=device)
    
    # === 关键修改：计算 pos_weight ===
    # 统计有多少个0（无边），有多少个1（有边）
    n_total = adj_label.numel()
    n_ones = adj_label.sum().item()
    n_zeros = n_total - n_ones
    
    # 权重 = 负样本数量 / 正样本数量
    # 这样 Loss 里的 1 就会被放大，模型必须学会预测 1
    pos_weight = torch.tensor([n_zeros / n_ones]).to(device)
    
    print(f"Graph Sparsity: Total={n_total}, Ones={n_ones}, Weight={pos_weight.item():.2f}")

    # 实例化模型
    model = GAE(n_input, n_hidden, n_z).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # === 将权重传入 Loss ===
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Start pre-training GAE on {dataset_name}...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward
        adj_logits, z = model(feat, adj_norm)
        
        # Loss 计算
        loss = criterion(adj_logits.view(-1), adj_label.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    # 保存权重
    save_path = f'./DCRN/model_pretrain/{dataset_name}_gae_pretrain.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"GAE weights saved to: {save_path}")
if __name__ == "__main__":
    pretrain_gae()