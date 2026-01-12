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
# 0. 辅助函数：实例级对比损失
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
        self.dataset = 'dblp'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # === 核心修改：加大预训练轮数，确保收敛 ===
        self.epochs_step1 = 160  # 生成视图多练一会儿
        self.epochs_step2 = 160 
        self.epochs_step3 = 160 
        self.lr = 1e-3
        
        self.save_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        if not os.path.exists('./model_pretrain'):
            os.makedirs('./model_pretrain')

args = PretrainArgs()

def main():
    # --- A. 数据加载 ---
    print(f">> Loading data for {args.dataset}...")
    adj, feat, label, adj_label = load_graph_data(
        args.dataset, 
        path='./DCRN/dataset/', 
        use_pca=False, 
        device=args.device
    )
    
    adj_sparse = adj.to_sparse().to(args.device)
    
    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    
    # === 核心修改：对齐 train_fusion.py 的宽网络配置 ===
    hidden_dim = 512  # 原来是 256
    z_dim = 128        # 提升到 128 (从 64) 增加表达能力
    gae_dims = [n_input, hidden_dim, z_dim] 
    
    print(f">> Pretrain Config: Input={n_input}, Hidden={hidden_dim}, Z={z_dim}")
    
    # --- B. 模型初始化 ---
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=n_input,
        hidden_dim=hidden_dim, # 这里也要改
        num_clusters=n_clusters,
        gae_dims=gae_dims
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss 权重准备
    pos_weight_val = float(adj_label.shape[0]**2 - adj_label.sum()) / adj_label.sum()
    norm_val = adj_label.shape[0]**2 / float((adj_label.shape[0]**2 - adj_label.sum()) * 2)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=args.device)

    def compute_recon_loss(adj_logits):
        return norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), adj_label.view(-1), pos_weight=pos_weight
        )

    def compute_kl_loss(mu, logstd):
        return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

    # === Step 1: VGAE ===
    print("\n=== Step 1: Pretraining Generative View ===")
    for epoch in range(args.epochs_step1):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        loss = compute_recon_loss(out['adj_logits']) + compute_kl_loss(out['mu'], out['logstd'])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 1 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # === Step 2: Denoising ===
    print("\n=== Step 2: Pretraining Denoising View (强化对比学习) ===")
    for epoch in range(args.epochs_step2):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        z_gen = out['mu'] 
        z_den = out['z_den'] 
        loss = contrastive_loss(z_gen, z_den) + 1e-4 * out['l0_loss']
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 2 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # === Step 3: Joint ===
    print("\n=== Step 3: Joint Pretraining (多视图对齐) ===")
    for epoch in range(args.epochs_step3):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        loss = compute_recon_loss(out['adj_logits']) + \
               compute_kl_loss(out['mu'], out['logstd']) + \
               contrastive_loss(out['mu'], out['z_den']) + \
               1e-4 * out['l0_loss']
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 3 | Epoch {epoch} | Loss: {loss.item():.4f}")

    print(f"\n>> Saving pretrained model to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()