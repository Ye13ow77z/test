import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# 复用 DCRN 的 GCNLayer 或定义一个标准的 GraphConv
class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft)
        self.act = nn.PReLU() # AdaGCL 常用 PReLU
    
    def forward(self, x, adj):
        out = self.fc(x)
        out = torch.spmm(adj, out)
        return self.act(out)

# === 1. 完整版 VGAE 生成器 ===
class GraphGenerativeModel(nn.Module):
    def __init__(self, n_input, n_z, n_layers=2):
        super(GraphGenerativeModel, self).__init__()
        # Encoder: GCNs
        self.gcn_layers = nn.ModuleList([GCNLayer(n_input, n_z) if i==0 else GCNLayer(n_z, n_z) for i in range(n_layers)])
        
        # 均值和方差层 (关键设计: 变分推断)
        self.encoder_mean = nn.Linear(n_z, n_z)
        self.encoder_std = nn.Sequential(nn.Linear(n_z, n_z), nn.Softplus()) # Softplus 保证方差为正

    def forward(self, x, adj):
        # 1. GCN 编码
        h = x
        for layer in self.gcn_layers:
            h = layer(h, adj)
            
        # 2. 计算分布参数
        z_mean = self.encoder_mean(h)
        z_std = self.encoder_std(h)
        
        # 3. 重参数化采样 (Reparameterization Trick)
        noise = torch.randn_like(z_std)
        z = z_mean + noise * z_std
        
        # 4. 解码生成图 (Inner Product Decoder)
        # 增加 Sigmoid 得到概率矩阵
        a_logits = torch.mm(z, z.t())
        a_probs = torch.sigmoid(a_logits)
        
        return a_probs, z_mean, z_std

# === 2. 完整版 去噪生成器 (含 Hard Concrete 采样) ===
class GraphDenoisingModel(nn.Module):
    def __init__(self, n_input, n_hid, gamma=-0.1, zeta=1.1):
        super(GraphDenoisingModel, self).__init__()
        # 参数初始化
        self.gamma = gamma
        self.zeta = zeta
        
        # 双线性注意力网络 (保留 AdaGCL 的交互设计)
        self.W_left = nn.Linear(n_input, n_hid)
        self.W_right = nn.Linear(n_input, n_hid)
        self.att_layer = nn.Linear(2 * n_hid, 1) # 预测边的 Log-Alpha
        
        self.edge_weights = [] # 用于存 L0 Loss

    def hard_concrete_sample(self, log_alpha, training=True):
        # 关键设计: Gumbel-Softmax 的变体，允许输出精确的 0 和 1
        if training:
            bias = 0.0
            random_noise = torch.rand_like(log_alpha)
            # Gumbel 噪声
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / 1.0 # beta=1.0
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)
            
        # 拉伸与截断 (Stretched and Rectified)
        stretched = gate_inputs * (self.zeta - self.gamma) + self.gamma
        clipped = torch.clamp(stretched, 0.0, 1.0)
        return clipped

    def l0_norm(self, log_alpha):
        # 计算稀疏正则 Loss
        reg = torch.sigmoid(log_alpha - 1.0 * np.log(-self.gamma / self.zeta))
        return torch.mean(reg)

    def generate(self, x, adj, training=True):
        # 1. 计算节点特征
        h_l = F.relu(self.W_left(x))
        h_r = F.relu(self.W_right(x))
        
        # 2. 仅对存在的边计算权重 (避免 N^2 复杂度)
        # 获取边的索引
        if adj.is_sparse:
            indices = adj._indices()
        else:
            indices = adj.nonzero().t()
            
        row, col = indices[0], indices[1]
        
        # 拼接特征并预测 log_alpha (边权参数)
        edge_h = torch.cat([h_l[row], h_r[col]], dim=1)
        log_alpha = self.att_layer(edge_h).squeeze()
        
        # 3. 采样 Mask
        mask = self.hard_concrete_sample(log_alpha, training)
        
        # 4. 保存用于计算 L0 Loss 的参数
        self.log_alpha = log_alpha 
        
        # 5. 生成去噪图 (Mask * Adj)
        # 构造新的稀疏矩阵
        if adj.is_sparse:
            values = adj._values() * mask
            new_adj = torch.sparse.FloatTensor(indices, values, adj.shape)
        else:
            # 稠密情况
            mask_mat = torch.zeros_like(adj)
            mask_mat[row, col] = mask
            new_adj = adj * mask_mat
            
        return new_adj