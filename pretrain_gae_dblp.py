import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import scipy.sparse as sp

# === 1. 工具函数 ===
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return torch.FloatTensor(adj_normalized.toarray())

# === 2. GAE ===
class GAE_AdaGCL_Style(nn.Module):
    def __init__(self, n_input, n_hidden, n_z):
        super(GAE_AdaGCL_Style, self).__init__()
        
        # 第一部分：GCN Backbone (Input -> Hidden)
        self.gcn_shared = nn.Linear(n_input, n_hidden)
        nn.init.xavier_uniform_(self.gcn_shared.weight)
        
        # 第二部分：MLP Head (Hidden -> Hidden -> Z)
        # 这就是 AdaGCL 的特征：用 MLP 而不是单层 GCN 来生成 Z
        self.encoder_mean = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_z)
        )
        # 初始化 MLP
        for layer in self.encoder_mean:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, adj):
        # 1. GCN 编码
        hidden = F.relu(torch.mm(adj, self.gcn_shared(x)))
        
        # 2. MLP 投影 (注意：这里是否乘 adj 取决于 AdaGCL 原文)
        # AdaGCL 原文的 encoder_mean 是纯 MLP，不乘 adj。
        # 但 VGAE 通常需要聚合。为了保持 AdaGCL 原味，我们这里只对 hidden 做 MLP
        z = self.encoder_mean(hidden)
        
        # Decoder (重构邻接矩阵)
        adj_logits = torch.matmul(z, z.t())
        return adj_logits, z

# === 3. 加载 DBLP 数据 ===
def load_data(dataset_name='dblp', path='./DCRN/dataset/'):
    print(f"Loading {dataset_name}...")
    root = os.path.join(path, dataset_name)
    feat = np.load(os.path.join(root, f'{dataset_name}_feat.npy'))
    adj = np.load(os.path.join(root, f'{dataset_name}_adj.npy'))
    adj_norm = normalize_adj(adj)
    feat = torch.FloatTensor(feat)
    adj_label = torch.FloatTensor(adj)
    return adj_norm, feat, adj_label

# === 4. 主训练流程 ===
if __name__ == "__main__":
    dataset = 'dblp'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    adj_norm, feat, adj_label = load_data(dataset)
    adj_norm, feat, adj_label = adj_norm.to(device), feat.to(device), adj_label.to(device)
    
    n_input = feat.shape[1]
    print(f"Dataset: {dataset} | Input Dim: {n_input} (AdaGCL Style Pretrain)")
    
    # 模型定义
    model = GAE_AdaGCL_Style(n_input=n_input, n_hidden=256, n_z=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Pos Weight
    n_total = adj_label.numel()
    n_ones = adj_label.sum().item()
    pos_weight = torch.tensor([(n_total - n_ones) / n_ones]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 训练
    for epoch in range(200):
        optimizer.zero_grad()
        logits, _ = model(feat, adj_norm)
        loss = criterion(logits.view(-1), adj_label.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # 保存权重
    save_path = f'./DCRN/model_pretrain/{dataset}_adagcl_pretrain.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved AdaGCL-style weights to {save_path}")