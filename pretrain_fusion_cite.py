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
# 0. 辅助 Loss 函数
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
        self.dataset = 'cite'  # 使用 cite 文件夹 (npy格式)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # === 策略调整 ===
        # 增加 Epoch，确保特征重构充分
        self.epochs_step1 = 200  
        self.epochs_step2 = 100 
        self.epochs_step3 = 200
        self.lr = 1e-3
        
        self.save_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        if not os.path.exists('./model_pretrain'):
            os.makedirs('./model_pretrain')

        self.hidden_dim = 512
        self.z_dim = 128

args = PretrainArgs()

def main():
    # --- A. 数据加载 (使用 load_graph_data 加载 npy 文件) ---
    print(f">> Loading data for {args.dataset}...")
    adj_sparse, feat, label, adj_label = load_graph_data(
        args.dataset, 
        path='./DCRN/dataset/', 
        use_pca=False, 
        device=args.device
    )

    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    num_nodes = feat.shape[0]
    gae_dims = [n_input, args.hidden_dim, args.z_dim]
    
    print(f">> Pretrain Config: Nodes={num_nodes}, Input={n_input}, Hidden={args.hidden_dim}, Z={args.z_dim}, Clusters={n_clusters}")

    # --- B. 模型初始化 ---
    model = AdaDCRN_VGAE(
        num_nodes=num_nodes,
        input_dim=n_input,
        hidden_dim=args.hidden_dim, 
        num_clusters=n_clusters,
        gae_dims=gae_dims,
        use_cluster_proj=False
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss 权重
    pos_sum = adj_label.sum().item()
    n_nodes_sq = adj_label.shape[0]**2
    pos_weight = float(n_nodes_sq - pos_sum) / (pos_sum + 1e-15)
    norm_val = n_nodes_sq / float((n_nodes_sq - pos_sum) * 2 + 1e-15)
    pos_weight_t = torch.as_tensor(pos_weight, dtype=torch.float32, device=args.device)

    def compute_recon_loss(adj_logits):
        return norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), adj_label.view(-1), pos_weight=pos_weight_t
        )

    def compute_kl_loss(mu, logstd):
        return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

    # ==========================================
    # Step 1: VGAE + 特征重构 (Dual Reconstruction)
    # 目标：既要学会连边(A)，也要学会复述内容(X)
    # ==========================================
    print("\n=== Step 1: Structure + Feature Reconstruction ===")
    for epoch in range(args.epochs_step1):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        # 1. 结构重构 (原有)
        loss_adj = compute_recon_loss(out['adj_logits'])
        loss_kl = compute_kl_loss(out['mu'], out['logstd'])
        
        # 2. 特征重构 (新增!!!) - 核心修正
        # 借用 den_decoder 来重构特征。我们需要保证 Z 里包含 X 的信息。
        # 这里我们将 mu (Generative View 的潜变量) 喂给 decoder 试图还原 feat
        # 核心修正：强制模型记住特征 X
        recon_feat = model.den_decoder(out['mu']) 
        
        # [修改点] 使用 BCEWithLogitsLoss，它对 0/1 特征更敏感，且梯度更大
        loss_feat = F.binary_cross_entropy_with_logits(recon_feat, feat)
        
        # [修改点] 不需要 10.0 的权重了，因为 BCE 本身就在 0.5~0.7 左右
        loss = loss_adj + loss_kl + 1.0 * loss_feat
        
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Step 1 | Epoch {epoch} | Loss: {loss.item():.4f} (Adj: {loss_adj.item():.2f}, Feat: {loss_feat.item():.4f})")

    # ==========================================
    # Step 2: Denoising View Pretraining
    # ==========================================
    print("\n=== Step 2: Denoising View (Contrastive) ===")
    for epoch in range(args.epochs_step2):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        z_gen = out['mu'].detach() 
        z_den = out['z_den'] 
        
        loss = contrastive_loss(z_gen, z_den) + 1e-4 * out['l0_loss']
        # Step 2 也加上特征重构辅助
        if 'recon_den' in out:
             loss += 10.0 * F.mse_loss(out['recon_den'], feat)

        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Step 2 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # ==========================================
    # Step 3: Joint Training
    # ==========================================
    print("\n=== Step 3: Joint Pretraining ===")
    for epoch in range(args.epochs_step3):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        loss_adj = compute_recon_loss(out['adj_logits'])
        loss_kl = compute_kl_loss(out['mu'], out['logstd'])
        loss_cl = contrastive_loss(out['mu'], out['z_den'])
        loss_l0 = 1e-4 * out['l0_loss']
        
        # 联合训练时继续保持特征重构约束
        # 联合训练时继续保持特征重构约束
        recon_feat = model.den_decoder(out['mu']) 
        
        # [修改点] 同样换成 BCE
        loss_feat = F.binary_cross_entropy_with_logits(recon_feat, feat)
        
        loss = loss_adj + loss_kl + loss_cl + loss_l0 + 1.0 * loss_feat
        
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Step 3 | Epoch {epoch} | Loss: {loss.item():.4f}")

    print(f"\n>> Saving pretrained model to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print(">> Pretraining Finished.")

if __name__ == "__main__":
    main()