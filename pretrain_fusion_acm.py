import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys

# 路径配置
sys.path.append(os.getcwd())

# 引入 ACM 专用加载器 (根据你 train_fusion.py 的引用)
try:
    from utils_acm import load_acm
except ImportError:
    print("Error: Could not import load_acm from utils_acm. Please check your files.")
    exit()

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
        self.dataset = 'acm'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ACM 规模较小，适当减少轮数防止过拟合，同时节省时间
        self.epochs_step1 = 120  # VGAE 重构
        self.epochs_step2 = 120  # 去噪对比
        self.epochs_step3 = 120  # 联合训练
        self.lr = 1e-3
        
        self.save_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        if not os.path.exists('./model_pretrain'):
            os.makedirs('./model_pretrain')

args = PretrainArgs()

def main():
    # --- A. 数据加载 (ACM 专用逻辑) ---
    print(f">> Loading data for {args.dataset}...")
    
    # 假设 ACM 文件在 ./DCRN/dataset/ACM3025.mat
    # load_acm 通常返回: adj(sparse), feat, label, adj_label(dense/weight)
    acm_path = './DCRN/dataset/ACM3025.mat'
    if not os.path.exists(acm_path):
        print(f"Error: File not found at {acm_path}")
        return

    # load_acm 内部已经处理了 sparse 转换，直接获取
    adj_sparse, feat, label, adj_label = load_acm(acm_path, args.device)
    
    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    
    # === 关键配置：必须与 train_fusion.py 保持一致 ===
    hidden_dim = 512   
    z_dim = 128        
    gae_dims = [n_input, hidden_dim, z_dim] 
    
    print(f">> Pretrain Config: Input={n_input}, Hidden={hidden_dim}, Z={z_dim}, Clusters={n_clusters}")
    
    # --- B. 模型初始化 ---
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=n_input,
        hidden_dim=hidden_dim, 
        num_clusters=n_clusters,
        gae_dims=gae_dims,
        use_cluster_proj=False 
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss 权重准备
    pos_sum = adj_label.sum().item()
    n_total = adj_label.shape[0]**2
    n_neg = n_total - pos_sum
    
    # 稳健的权重计算
    pos_weight_val = float(n_neg / (pos_sum + 1e-15))
    norm_val = n_total / float((n_neg * 2) + 1e-15)
    pos_weight = torch.as_tensor(pos_weight_val, dtype=torch.float32, device=args.device)

    def compute_recon_loss(adj_logits):
        return norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), adj_label.view(-1), pos_weight=pos_weight
        )

    def compute_kl_loss(mu, logstd):
        return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

    # === Step 1: VGAE 生成视图预热 ===
    print("\n=== Step 1: Pretraining Generative View (Reconstruction) ===")
    for epoch in range(args.epochs_step1):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        loss = compute_recon_loss(out['adj_logits']) + compute_kl_loss(out['mu'], out['logstd'])
        
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 1 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # === Step 2: DenoisingNet 去噪视图预热 ===
    print("\n=== Step 2: Pretraining Denoising View (Contrastive) ===")
    for epoch in range(args.epochs_step2):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        z_gen = out['mu'].detach() 
        z_den = out['z_den'] 
        
        loss = contrastive_loss(z_gen, z_den) + 1e-4 * out['l0_loss']
        
        if 'recon_den' in out:
             loss += 0.1 * F.mse_loss(out['recon_den'], feat)

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 2 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # === Step 3: Joint Training 联合微调 ===
    print("\n=== Step 3: Joint Pretraining (All Components) ===")
    for epoch in range(args.epochs_step3):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        loss_recon = compute_recon_loss(out['adj_logits'])
        loss_kl = compute_kl_loss(out['mu'], out['logstd'])
        loss_cl = contrastive_loss(out['mu'], out['z_den'])
        loss_l0 = 1e-4 * out['l0_loss']
        
        loss = loss_recon + loss_kl + loss_cl + loss_l0
        
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 3 | Epoch {epoch} | Loss: {loss.item():.4f}")

    print(f"\n>> Saving pretrained model to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print(">> Done.")

if __name__ == "__main__":
    main()