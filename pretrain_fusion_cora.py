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
# 0. 辅助函数：实例级对比损失 (SimCLR Style)
# ====================================================================
def contrastive_loss(z1, z2, temperature=0.5):
    # z1, z2: [N, D]
    # 标准化
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # 相似度矩阵
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    # 正样本对在对角线上
    pos_sim = torch.diag(sim_matrix)
    
    # Loss: -log( exp(pos) / sum(exp(all)) )
    loss = -torch.mean(torch.log_softmax(sim_matrix, dim=1).diag())
    return loss

# ====================================================================
# 1. 配置参数 (Cora 特供版)
# ====================================================================
class PretrainArgs:
    def __init__(self):
        # === 修改 1: 指定 Cora ===
        self.dataset = 'cora'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # === 修改 2: 针对 Cora 微调轮数 (Cora 较小，收敛快) ===
        self.epochs_step1 = 30   # 生成视图预训练 (VGAE Recon)
        self.epochs_step2 = 30   # 降噪视图预训练 (Contrastive)
        self.epochs_step3 = 50   # 联合微调
        self.lr = 1e-3
        
        # 保存路径
        self.save_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        if not os.path.exists('./model_pretrain'):
            os.makedirs('./model_pretrain')

args = PretrainArgs()

# ====================================================================
# 2. 预训练流程
# ====================================================================
def main():
    # --- A. 数据加载 ---
    print(f">> Loading data for {args.dataset}...")
    # 确保 load_graph_data 能自动处理 cora (通常标准代码里有)
    adj, feat, label, adj_label = load_graph_data(
        args.dataset, 
        path='./DCRN/dataset/', 
        use_pca=False, 
        device=args.device
    )
    
    # 转为稀疏张量
    adj_sparse = adj.to_sparse().to(args.device)
    
    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    # 保持和 train_fusion.py 一致的维度结构
    gae_dims = [n_input, 256, 50] 
    
    print(f">> Input Dim: {n_input}, Clusters: {n_clusters}")
    
    # --- B. 模型初始化 ---
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=n_input,
        hidden_dim=256,
        num_clusters=n_clusters,
        gae_dims=gae_dims
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 重构 Loss 权重计算 (处理正负样本不平衡)
    pos_weight_val = float(adj_label.shape[0]**2 - adj_label.sum()) / adj_label.sum()
    norm_val = adj_label.shape[0]**2 / float((adj_label.shape[0]**2 - adj_label.sum()) * 2)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=args.device)

    def compute_recon_loss(adj_logits):
        return norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), adj_label.view(-1), pos_weight=pos_weight
        )

    def compute_kl_loss(mu, logstd):
        return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

    # =================================================================
    # Step 1: 预训练生成视图 (VGAE)
    # =================================================================
    print(f"\n=== Step 1: Pretraining Generative View (VGAE) [{args.epochs_step1} Epochs] ===")
    for epoch in range(args.epochs_step1):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播 (此时只关注 view_gen)
        # 注意: 这里调用 model(feat, adj_sparse) 会跑全流程，
        # 但我们只取需要的 loss，梯度会自动回传到相关部分
        out = model(feat, adj_sparse)
        
        loss_recon = compute_recon_loss(out['adj_logits'])
        loss_kl = compute_kl_loss(out['mu'], out['logstd'])
        
        loss = loss_recon + loss_kl
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Step 1 | Epoch {epoch:02d} | Loss: {loss.item():.4f} (Recon: {loss_recon.item():.4f})")

    # =================================================================
    # Step 2: 预训练降噪视图 (Denoising View)
    # =================================================================
    print(f"\n=== Step 2: Pretraining Denoising View [{args.epochs_step2} Epochs] ===")
    for epoch in range(args.epochs_step2):
        model.train()
        optimizer.zero_grad()
        
        out = model(feat, adj_sparse)
        
        # 获取特征
        z_gen = out['mu']      # 生成视图特征 (Anchor)
        z_den = out['z_den']   # 降噪视图特征 (Positive)
            
        # 1. 实例级对比损失 (拉近 Gen 和 Den 的特征)
        loss_cl = contrastive_loss(z_gen, z_den)
        
        # 2. L0 Loss (控制稀疏度，防止去噪网络偷懒全置0或全置1)
        loss_l0 = out['l0_loss']
        
        loss = loss_cl + 0.0 * loss_l0
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Step 2 | Epoch {epoch:02d} | Loss: {loss.item():.4f} (CL: {loss_cl.item():.4f})")

    # =================================================================
    # Step 3: 联合预训练 (Joint Training)
    # =================================================================
    print(f"\n=== Step 3: Joint Pretraining [{args.epochs_step3} Epochs] ===")
    for epoch in range(args.epochs_step3):
        model.train()
        optimizer.zero_grad()
        
        out = model(feat, adj_sparse)
        
        z_gen = out['mu']
        z_den = out['z_den']
            
        loss_recon = compute_recon_loss(out['adj_logits'])
        loss_kl = compute_kl_loss(out['mu'], out['logstd'])
        loss_cl = contrastive_loss(z_gen, z_den)
        loss_l0 = out['l0_loss']
        
        # 综合 Loss
        loss = loss_recon + loss_kl + loss_cl + 1e-4 * loss_l0
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Step 3 | Epoch {epoch:02d} | Total Loss: {loss.item():.4f}")

    # --- C. 保存权重 ---
    print(f"\n>> Saving pretrained model to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print("Done! You can now run train_fusion.py with dataset='cora'.")

if __name__ == "__main__":
    main()