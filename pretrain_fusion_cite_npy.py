import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys

# 路径配置
sys.path.append(os.getcwd())

from utils_data import load_graph_data
from model_fusion import AdaDCRN_VGAE

# ====================================================================
# 0. 辅助函数
# ====================================================================
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    loss = -torch.mean(torch.log_softmax(sim_matrix, dim=1).diag())
    return loss

# ====================================================================
# 1. 配置参数
# ====================================================================
class PretrainArgs:
    def __init__(self):
        # 对应文件夹名，例如 ./DCRN/dataset/cite/
        self.dataset = 'cite'  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 预训练轮数 (加长预训练，让特征充分学习)
        self.epochs_step1 = 300  # VGAE 重构预热 (加长)
        self.epochs_step2 = 150  # 去噪视图对比预热 (加长)
        self.epochs_step3 = 300  # 联合预训练 (加长)
        self.lr = 5e-4           # 降低学习率，更稳定
        
        self.save_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        if not os.path.exists('./model_pretrain'):
            os.makedirs('./model_pretrain')

        # 核心维度配置
        self.hidden_dim = 512
        self.z_dim = 128

args = PretrainArgs()

def main():
    # --- A. 数据加载 (NPY 模式) ---
    print(f">> Loading data for {args.dataset}...")
    
    # 使用通用加载器读取 .npy 文件
    # 路径会自动寻找 ./DCRN/dataset/cite/cite_feat.npy 等
    adj_norm, feat, label, adj_label = load_graph_data(
        args.dataset, 
        path='./DCRN/dataset/', 
        use_pca=False, 
        device=args.device
    )
    
    # 转换为稀疏张量用于 GCN 输入
    if isinstance(adj_norm, torch.Tensor) and adj_norm.is_sparse:
        adj_sparse = adj_norm.to(args.device)
    else:
        adj_sparse = adj_norm.to_sparse().to(args.device)
    
    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    gae_dims = [n_input, args.hidden_dim, args.z_dim]
    
    print(f">> Pretrain Config: Input={n_input}, Hidden={args.hidden_dim}, Z={args.z_dim}, Clusters={n_clusters}")
    
    # --- B. 模型初始化 ---
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=n_input,
        hidden_dim=args.hidden_dim, 
        num_clusters=n_clusters,
        gae_dims=gae_dims,
        use_cluster_proj=False # 预训练不训练聚类头
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss 权重准备 (用于重构邻接矩阵)
    # adj_label 是 Dense Tensor (来自 utils_data)
    pos_sum = adj_label.sum()
    n_nodes_sq = adj_label.shape[0]**2
    pos_weight_val = float(n_nodes_sq - pos_sum) / (pos_sum + 1e-15)
    norm_val = n_nodes_sq / float((n_nodes_sq - pos_sum) * 2 + 1e-15)
    pos_weight = torch.as_tensor(pos_weight_val, dtype=torch.float32, device=args.device)

    def compute_recon_loss(adj_logits):
        return norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), adj_label.view(-1), pos_weight=pos_weight
        )

    def compute_kl_loss(mu, logstd):
        return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

    # ==========================================
    # Step 1: VGAE 生成视图预热
    # ==========================================
    print("\n=== Step 1: Pretraining Generative View (Reconstruction) ===")
    for epoch in range(args.epochs_step1):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        # 1. 结构重构 (Adj)
        loss_adj = compute_recon_loss(out['adj_logits'])
        loss_kl = compute_kl_loss(out['mu'], out['logstd'])
        
        # 2. [Citeseer 特供] 特征重构 (MSE)
        # 这一步在 DBLP 里没有，但对 Citeseer 至关重要，否则 ACC 会崩到 30%
        # 我们手动调用 den_decoder 来重构 mu (即 z_gen)
        recon_feat = model.den_decoder(out['mu']) 
        loss_feat = F.mse_loss(recon_feat, feat)
        
        # 权重 20.0 强迫模型记住特征 (cite特征是0/1稀疏的，需要更强约束)
        loss = loss_adj + loss_kl + 20.0 * loss_feat
        
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 1 | Epoch {epoch} | Loss: {loss.item():.4f} (Adj: {loss_adj.item():.2f}, Feat: {loss_feat.item():.4f})")

    # ==========================================
    # Step 2: DenoisingNet 去噪视图预热
    # ==========================================
    print("\n=== Step 2: Pretraining Denoising View (Contrastive) ===")
    for epoch in range(args.epochs_step2):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        z_gen = out['mu'].detach() 
        z_den = out['z_den'] 
        
        # 对比损失 + L0 正则
        loss = contrastive_loss(z_gen, z_den) + 1e-4 * out['l0_loss']
        
        # [Citeseer 特供] 辅助特征重构
        if 'recon_den' in out:
             loss += 20.0 * F.mse_loss(out['recon_den'], feat)

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 2 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # ==========================================
    # Step 3: Joint Training 联合微调
    # ==========================================
    print("\n=== Step 3: Joint Pretraining (All Components) ===")
    for epoch in range(args.epochs_step3):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        loss_adj = compute_recon_loss(out['adj_logits'])
        loss_kl = compute_kl_loss(out['mu'], out['logstd'])
        loss_cl = contrastive_loss(out['mu'], out['z_den'])
        loss_l0 = 1e-4 * out['l0_loss']
        
        # [Citeseer 特供] 联合训练时保持特征约束
        recon_feat = model.den_decoder(out['mu']) 
        loss_feat = F.mse_loss(recon_feat, feat)
        
        loss = loss_adj + loss_kl + loss_cl + loss_l0 + 20.0 * loss_feat
        
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 3 | Epoch {epoch} | Loss: {loss.item():.4f}")

    print(f"\n>> Saving pretrained model to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print(">> Done.")

if __name__ == "__main__":
    main()