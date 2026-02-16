import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

# ====================================================================
# 0. 适配层 (Global Args Proxy)
# ====================================================================
class GlobalArgs:
    def __init__(self):
        self.user = 0
        self.item = 0
        self.latdim = 256
        self.gnn_layer = 2
        self.temp = 0.2
        self.gamma = -0.5
        self.zeta = 1.1
        self.lambda0 = 1e-4 # L0 loss weight
        self.reg = 1e-4

args = GlobalArgs()

# 初始化函数
init = nn.init.xavier_uniform_

# ====================================================================
# 1. 基础组件
# ====================================================================

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        # 1. Linear Transformation
        x = self.linear(x)
        # 2. Sparse Propagation
        # 兼容 torch.sparse 和 dense tensor
        if adj.is_sparse:
            out = torch.spmm(adj, x)
        else:
            out = torch.mm(adj, x)
        return F.relu(out)

class vgae_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(vgae_encoder, self).__init__()
        # 使用自定义 GCNLayer 替换原来的手动 spmm 实现，结构更清晰
        self.base_gcn = GCNLayer(input_dim, hidden_dim)
        
        # 均值和方差层
        self.encoder_mean = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(z_dim, z_dim)
        )
        self.encoder_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(z_dim, z_dim), 
            nn.Softplus()
        )

    def forward(self, x, adj):
        # Base GCN Layer
        hidden = self.base_gcn(x, adj)
        
        # Latent Variables
        x_mean = self.encoder_mean(hidden)
        x_std = self.encoder_std(hidden)
        
        # Reparameterization Trick
        if self.training:
            gaussian_noise = torch.randn_like(x_mean)
            z = gaussian_noise * x_std + x_mean
        else:
            z = x_mean
        return z, x_mean, x_std

class vgae_decoder(nn.Module):
    def __init__(self):
        super(vgae_decoder, self).__init__()
        # 移除了 MLP，统一使用 Inner Product Decoder
        # 这对于图聚类任务更标准，且保证训练和生成的一致性
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # 训练用：全图 Logits (N x N)
        # 注意：对于超大图可能需要负采样优化，但在 DBLP/ACM 规模通常可以直接算
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
        """
        生成视图逻辑：
        使用训练好的 Encoder 得到 Z，通过 Inner Product 计算边概率，
        并采样生成新的增强图结构。
        """
        # 1. 获取潜在表示
        with torch.no_grad():
            z, _, _ = self.encoder(x, adj)
        
        # 2. 仅计算现有边的概率 (Masked Attention 思想)
        # 为了保持稀疏性，我们不生成全连接图，而是对现有结构进行重加权/筛选
        indices = adj._indices()
        row, col = indices[0], indices[1]
        
        z_row = z[row]
        z_col = z[col]
        
        # Inner Product: (N_edges, D) * (N_edges, D) -> sum -> (N_edges,)
        edge_logits = (z_row * z_col).sum(dim=1)
        edge_probs = torch.sigmoid(edge_logits)
        
        # 3. 采样 Mask (保留概率 > 0.5 的边)
        # 也可以加入随机性： torch.bernoulli(edge_probs)
        mask = ((edge_probs + 0.5).floor()).type(torch.bool)
        
        # 4. 构造新图
        new_indices = indices[:, mask]
        new_values = adj._values()[mask]
        
        # 5. 归一化 (保持总边权能量守恒，可选)
        if new_values.sum() > 0:
            ratio = adj._values().sum() / (new_values.sum() + 1e-12)
            new_values = new_values * ratio
            
        new_adj = torch.sparse.FloatTensor(new_indices, new_values, adj.shape).coalesce()
        return new_adj

class DenoisingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingNet, self).__init__()
        
        # 对应 AdaGCL 的结构
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
        
        f1_features = x[row] 
        f2_features = x[col] 
        
        # 2. 计算 Attention 并保存用于 L0 Loss
        weight = self.get_attention(f1_features, f2_features)
        self.edge_weights = [weight] 
        
        # 3. 采样 Mask
        mask = self.hard_concrete_sample(weight, beta=1.0, training=training)
        mask = mask.squeeze()
        
        # 4. 构造去噪图 (Masked Values)
        masked_values = adj._values() * mask
        
        # --- 稀疏对称归一化 (Symmetric Normalization) ---
        
        # A: Coalesce 用于正确聚合重复索引（如果有）
        temp_adj = torch.sparse_coo_tensor(indices, masked_values, adj.shape, device=adj.device).coalesce()
        temp_indices = temp_adj._indices()
        temp_values = temp_adj._values()
        
        # B: 计算度 (Degree)
        rowsum = torch.sparse.sum(temp_adj, dim=1).to_dense() + 1e-10
        
        # C: 计算 D^-0.5
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        # [AdaGCL Fix] 增加数值截断，防止梯度爆炸
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0) 
        
        # D: 应用归一化
        temp_row, temp_col = temp_indices[0], temp_indices[1]
        norm_values = temp_values * d_inv_sqrt[temp_row] * d_inv_sqrt[temp_col]
        
        # E: 构造最终的归一化稀疏图
        adj_den_norm = torch.sparse_coo_tensor(temp_indices, norm_values, adj.shape, device=adj.device)
        
        return adj_den_norm

# ====================================================================
# 2. 融合与主模型 (AdaDCRN_VGAE)
# ====================================================================

class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False)
        )
        # 初始化第二层权重使得初始注意力较为平衡
        with torch.no_grad():
            self.att[2].weight.fill_(0.0)
        self.temperature = 0.50 

    def forward(self, z_list):
        # z_list: [z_gen, z_den] -> (N, 2, D)
        h = torch.stack(z_list, dim=1) 
        att_score = self.att(h) # (N, 2, 1)
        
        # Softmax over view dimension
        weights = F.softmax(att_score / self.temperature, dim=1) 
        
        z_global = torch.sum(h * weights, dim=1)
        return z_global, weights

class AdaDCRN_VGAE(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_clusters, gae_dims, use_cluster_proj=False):
        super(AdaDCRN_VGAE, self).__init__()
        
        # 注入参数到全局 args
        global args
        args.user = num_nodes
        args.latdim = hidden_dim
        
        z_dim = gae_dims[-1]
        
        # === 核心组件 ===
        
        # 1. 生成视图 (VGAE) - 使用 Inner Product Decoder
        self.encoder_vgae = vgae_encoder(input_dim, hidden_dim, z_dim)
        self.decoder_vgae = vgae_decoder() # 无参数
        self.view_gen = vgae(self.encoder_vgae, self.decoder_vgae)
        
        # 2. 去噪视图 (AdaGCL DenoisingNet)
        self.view_den = DenoisingNet(input_dim, hidden_dim)
        
        # 3. 融合层
        self.fusion = AttentionFusion(z_dim)
        
        # 4. 聚类投影与预测头
        self.use_cluster_proj = use_cluster_proj
        if self.use_cluster_proj:
            self.cluster_proj = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.PReLU(),
                nn.Linear(z_dim, z_dim)
            )
        else:
            self.cluster_proj = nn.Identity()
            
        self.head = nn.Linear(z_dim, num_clusters)
        
        # 5. 去噪视图的辅助重构解码器
        self.den_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, adj):
        # A. 生成视图流 (Generative View)
        # 获取 z_gen, 用于训练重构 loss
        z_gen, adj_logits, mu, logstd = self.view_gen(x, adj)
        
        # B. 去噪视图流 (Denoising View)
        # 1. 学习去噪并生成图
        adj_den = self.view_den.generate(x, adj, training=self.training)
        
        # 2. L0 Loss
        l0_loss = self.view_den.l0_norm(self.view_den.edge_weights[0])
        
        # 3. 编码去噪图 (共享/复用 VGAE Encoder)
        z_den, _, _ = self.view_gen.encoder(x, adj_den)
        
        # C. 融合 (Fusion)
        z_fused, weights = self.fusion([z_gen, z_den])
        
        # D. 聚类输出
        z_fused_proj = self.cluster_proj(z_fused)
        z_gen_proj = self.cluster_proj(z_gen)
        z_den_proj = self.cluster_proj(z_den)
        
        q_fused = F.softmax(self.head(z_fused_proj), dim=1)
        q_gen   = F.softmax(self.head(z_gen_proj), dim=1)
        q_den   = F.softmax(self.head(z_den_proj), dim=1)
        
        # 辅助重构 (如果需要对 z_den 做特征重构监督)
        recon_den = self.den_decoder(z_den)

        return {
            "q": q_fused,
            "q_gen": q_gen,
            "q_den": q_den,
            "adj_logits": adj_logits, # 对应 VGAE 的重构目标
            "z_fused": z_fused,
            "mu": mu,                 # 对应 VGAE 的 KL 目标
            "logstd": logstd,
            "l0_loss": l0_loss,
            "z_den": z_den,
            "att_weights": weights,
            "recon_den": recon_den,
            "feat": x
        }