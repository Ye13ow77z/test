import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# 基础组件
# ====================================================================

class DCRN_GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(DCRN_GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None: self.linear.bias.data.fill_(0.0)

    def forward(self, x, adj):
        h = self.linear(x)
        h = torch.spmm(adj, h)
        if self.activation is not None:
            h = self.activation(h)
        return h

# ====================================================================
# 视图 1: VGAE 生成视图 (变分图自编码器)
# ====================================================================

class AdaGCL_VGAE_View(nn.Module):
    def __init__(self, layer_dims):
        super(AdaGCL_VGAE_View, self).__init__()
        self.layers = nn.ModuleList()
        
        # 编码器前几层 (Shared GCN)
        # 假设 dims = [1433, 256, 50]
        # input -> hidden (1433 -> 256)
        for i in range(len(layer_dims) - 2):
            self.layers.append(DCRN_GCNLayer(layer_dims[i], layer_dims[i+1], nn.Tanh()))
            
        # 最后一层拆分为两个 Head: Mu 和 LogStd
        hidden_dim = layer_dims[-2]
        z_dim = layer_dims[-1]
        
        self.gc_mu = DCRN_GCNLayer(hidden_dim, z_dim, activation=None)
        self.gc_logstd = DCRN_GCNLayer(hidden_dim, z_dim, activation=None)

    def reparameterize(self, mu, logstd):
        """VGAE 重参数化技巧: z = mu + sigma * eps"""
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
            
        # 计算均值和对数标准差
        mu = self.gc_mu(h, adj)
        logstd = self.gc_logstd(h, adj)
        
        # 采样得到 Z
        z = self.reparameterize(mu, logstd)
        
        # 解码 (生成 Logits)
        adj_logits = torch.mm(z, z.t())
        
        return z, adj_logits, mu, logstd

# ====================================================================
# 视图 2: 降噪视图 (保持不变，用于提供对比特征)
# ====================================================================

class AdaGCL_Denoising_View(nn.Module):
    def __init__(self, layer_dims):
        super(AdaGCL_Denoising_View, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            act = nn.Tanh() if i < len(layer_dims) - 2 else None
            self.layers.append(DCRN_GCNLayer(layer_dims[i], layer_dims[i+1], act))
            
    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        z = h
        return z, None

# ====================================================================
# 融合与主模型 (适配 VGAE)
# ====================================================================

class DCRN_Fusion(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super(DCRN_Fusion, self).__init__()
        self.a = nn.Parameter(torch.ones(num_nodes, hidden_dim) * 0.5)
        self.b = nn.Parameter(torch.ones(num_nodes, hidden_dim) * 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, z1, z2, adj):
        z_i = self.a * z1 + self.b * z2
        z_l = torch.spmm(adj, z_i)
        z_fused = self.alpha * z_l + (1 - self.alpha) * z_i
        return z_fused

class AdaDCRN_VGAE(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_clusters, gae_dims):
        super(AdaDCRN_VGAE, self).__init__()
        z_dim = gae_dims[-1]
        
        # 使用 VGAE 视图
        self.view_gen = AdaGCL_VGAE_View(gae_dims)
        self.view_den = AdaGCL_Denoising_View(gae_dims)
        self.fusion = DCRN_Fusion(num_nodes, z_dim)
        
        self.head = nn.Linear(z_dim, num_clusters)

    def forward(self, x, adj):
        # Generative View 返回: z, logits, mu, logstd
        z_gen, adj_logits, mu, logstd = self.view_gen(x, adj)
        
        # Denoising View
        z_den, _ = self.view_den(x, adj)
        
        # Fusion
        z_fused = self.fusion(z_gen, z_den, adj)
        
        # Clustering Head
        q = F.softmax(self.head(z_fused), dim=1)
        
        return {
            "q": q, 
            "adj_logits": adj_logits,
            "z_fused": z_fused,
            "z_gen": z_gen,
            "mu": mu,          # VGAE 需要用来算 KL Loss
            "logstd": logstd   # VGAE 需要用来算 KL Loss
        }