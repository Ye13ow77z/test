import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.sparse as sp
import math

# ====================================================================
# 0. 适配层 (Global Args Proxy)
# ====================================================================
# 不修改 AdaGCL 源码中的 args.x 调用，定义一个全局代理
class GlobalArgs:
    def __init__(self):
        self.user = 0
        self.item = 0
        self.latdim = 256
        self.gnn_layer = 2
        self.temp = 0.2
        self.gamma = -0.1
        self.zeta = 1.1
        self.lambda0 = 1e-4 # L0 loss weight
        self.reg = 1e-4

args = GlobalArgs()

# 初始化函数
init = nn.init.xavier_uniform_

# ====================================================================
# 1. 原始 AdaGCL 组件 (Original Components from AdaGCL/Model.py)
# ====================================================================

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds, flag=True):
        # 适配: DBLP 数据通常是 torch.sparse.FloatTensor
        # AdaGCL 原版处理 SparseTensor，这里做兼容
        if (flag):
            return torch.spmm(adj, embeds)
        else:
            # 如果传入的是 coalesced sparse tensor
            if adj.is_sparse:
                return torch.spmm(adj, embeds)
            else:
                return torch.mm(adj, embeds)

class vgae_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(vgae_encoder, self).__init__()
        # 这里的 hidden 对应 AdaGCL 的 latdim
        # 我们稍微修改 __init__ 以支持动态维度，但保持 forward 逻辑不变
        self.gcn_shared = nn.Linear(input_dim, hidden_dim) # 模拟 forward_graphcl 的 GCN
        
        self.encoder_mean = nn.Sequential(nn.Linear(hidden_dim, z_dim), nn.ReLU(inplace=True), nn.Linear(z_dim, z_dim))
        self.encoder_std = nn.Sequential(nn.Linear(hidden_dim, z_dim), nn.ReLU(inplace=True), nn.Linear(z_dim, z_dim), nn.Softplus())
        
        # 初始化
        nn.init.xavier_uniform_(self.gcn_shared.weight)

    def forward(self, x, adj):
        # 1. GCN 编码 (模拟 forward_graphcl)
        # AdaGCL 原版是用多层 GCN，这里我们用一层或两层来提取特征
        hidden = F.relu(torch.spmm(adj, self.gcn_shared(x)))
        
        # 2. VGAE Head
        x_mean = self.encoder_mean(hidden)
        x_std = self.encoder_std(hidden)
        
        # 3. Reparameterization
        if self.training:
            gaussian_noise = torch.randn_like(x_mean)
            z = gaussian_noise * x_std + x_mean
        else:
            z = x_mean
            
        return z, x_mean, x_std

class vgae_decoder(nn.Module):
    def __init__(self, hidden=256):
        super(vgae_decoder, self).__init__()
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # 原版是用 users/items 索引做点积，这里做全图重构 (Dot Product)
        adj_logits = torch.mm(z, z.t())
        return adj_logits

class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, adj):
        z, x_mean, x_std = self.encoder(x, adj)
        adj_logits = self.decoder(z)
        return z, adj_logits, x_mean, x_std

    def generate(self, x, adj):
        # AdaGCL 的生成逻辑：用 decoder 生成概率，然后采样 mask
        z, _, _ = self.encoder(x, adj)
        
        # 这里的逻辑是将 Z 经过 Decoder MLP 得到边权重
        # 为了简化全图计算，我们只采样存在的边 (Indices)
        indices = adj._indices()
        row, col = indices[0], indices[1]
        
        z_row = z[row]
        z_col = z[col]
        
        # Decoder 输出 logits
        edge_logits = self.decoder.decoder(z_row * z_col).squeeze()
        edge_probs = self.sigmoid(edge_logits)
        
        # 采样 Mask (保留概率 > 0.5 的边)
        mask = ((edge_probs + 0.5).floor()).type(torch.bool)
        
        # 构造新图
        new_indices = indices[:, mask]
        new_values = adj._values()[mask]
        
        # 归一化 (保持总边权与原图一致)
        if new_values.sum() > 0:
            new_values = new_values * (adj._values().sum() / new_values.sum())
            
        new_adj = torch.sparse.FloatTensor(new_indices, new_values, adj.shape).coalesce()
        return new_adj

class DenoisingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingNet, self).__init__()
        
        # 对应 AdaGCL 的 nblayers 和 selflayers
        self.nblayers_0 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True))
        self.selflayers_0 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True))
        
        # Attention 权重层
        self.attentions_0 = nn.Sequential(nn.Linear(2 * hidden_dim, 1))
        
        self.edge_weights = []

    def get_attention(self, input1, input2):
        input1 = self.nblayers_0(input1) # Neighbor features
        input2 = self.selflayers_0(input2) # Self features
        
        input10 = torch.cat([input1, input2], dim=1)
        weight10 = self.attentions_0(input10)
        return weight10

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        gamma = args.gamma
        zeta = args.zeta

        if training:
            debug_var = 1e-7
            bias = 0.0
            # 随机噪声
            random_noise = torch.rand_like(log_alpha) + debug_var
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = torch.clamp(stretched_values, 0.0, 1.0)
        return cliped.float()

    def l0_norm(self, log_alpha, beta=1.0):
        gamma = args.gamma
        zeta = args.zeta
        # 转 tensor 防止报错
        gamma_t = torch.tensor(gamma, device=log_alpha.device)
        zeta_t = torch.tensor(zeta, device=log_alpha.device)
        
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma_t/zeta_t))
        return torch.mean(reg_per_weight)

    def generate(self, x, adj, training=True):
        # 1. 准备特征
        indices = adj._indices()
        row, col = indices[0], indices[1]
        
        f1_features = x[row] # Source Node Features
        f2_features = x[col] # Target Node Features
        
        # 2. 计算 Attention (Log Alpha)
        weight = self.get_attention(f1_features, f2_features)
        self.edge_weights = [weight] # 存起来算 Loss
        
        # 3. Hard Concrete Sample 得到 Mask
        mask = self.hard_concrete_sample(weight, beta=1.0, training=training)
        mask = mask.squeeze()
        
        # 4. 构造去噪后的图
        new_values = adj._values() * mask
        new_adj = torch.sparse.FloatTensor(indices, new_values, adj.shape).coalesce()
        
        # 归一化 (GCN 需要)
        # 简单的对称归一化逻辑
        rowsum = torch.sparse.sum(new_adj, dim=1).to_dense() + 1e-6
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        # D^-0.5 * A * D^-0.5 (Dense 近似，稀疏乘法太慢)
        # 为了效率，这里直接返回 new_adj，假设 GCN 内部能处理或外部处理
        return new_adj

# ====================================================================
# 2. 融合与主模型 (AdaDCRN_VGAE)
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
        
        # 注入参数到全局 args，让 AdaGCL 组件能读取
        global args
        args.user = num_nodes # DBLP 节点数映射为 User 数
        args.item = 0         # 没有 Item
        args.latdim = hidden_dim
        args.gnn_layer = 2
        
        z_dim = gae_dims[-1]
        
        # === 核心组件实例化 ===
        
        # 1. 生成视图 (VGAE)
        self.encoder_vgae = vgae_encoder(input_dim, hidden_dim, z_dim)
        self.decoder_vgae = vgae_decoder(z_dim)
        self.view_gen = vgae(self.encoder_vgae, self.decoder_vgae)
        
        # 2. 去噪视图 (AdaGCL DenoisingNet)
        # 这里用 hidden_dim 作为内部维度
        self.view_den = DenoisingNet(input_dim, hidden_dim)
        
        # 3. 融合层
        self.fusion = DCRN_Fusion(num_nodes, z_dim)
        
        # 4. 聚类头
        self.head = nn.Linear(z_dim, num_clusters)

    def forward(self, x, adj):
        # A. 生成视图流 (Generative View)
        # 1. 生成增强图 (Augmented Graph)
        # 注意: generate 会调用 encoder 和 decoder 来采样边
        # adj_gen = self.view_gen.generate(x, adj) 
        # 为了效率和梯度流，我们直接用 encoder 得到的 z_gen，不需要显式构建 adj_gen 再 encode 一遍
        # 除非是为了做 GraphCL 对比。这里我们直接取 VGAE 的 Latent Z 作为 View 1 的特征
        z_gen, adj_logits, mu, logstd = self.view_gen(x, adj)
        
        # B. 去噪视图流 (Denoising View)
        # 1. 学习去噪掩码并生成去噪图 (Learnable Mask)
        adj_den = self.view_den.generate(x, adj, training=self.training)
        
        # 2. 计算 L0 Regularization Loss (用于稀疏化)
        l0_loss = self.view_den.l0_norm(self.view_den.edge_weights[0])
        
        # 3. 编码去噪图
        # 复用 view_gen 的 encoder (Shared Encoder 策略)
        z_den, _, _ = self.view_gen.encoder(x, adj_den)
        
        # C. 融合 (Fusion)
        z_fused = self.fusion(z_gen, z_den, adj)
        
        # D. 聚类输出 (En-CLU)
        q_fused = F.softmax(self.head(z_fused), dim=1)
        q_gen   = F.softmax(self.head(z_gen), dim=1)
        q_den   = F.softmax(self.head(z_den), dim=1)
        
        return {
            "q": q_fused,
            "q_gen": q_gen,
            "q_den": q_den,
            "adj_logits": adj_logits, # 重构 Loss 用
            "z_fused": z_fused,
            "mu": mu,                 # VGAE KL Loss 用
            "logstd": logstd,
            "l0_loss": l0_loss        # 加入总 Loss
        }