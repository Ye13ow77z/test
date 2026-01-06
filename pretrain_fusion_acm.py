import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

# 路径配置
sys.path.append(os.getcwd())

from model_fusion import AdaDCRN_VGAE
# === 引入新的加载器 ===
from utils_acm import load_acm 

# ====================================================================
# 配置参数 (ACM 特供版)
# ====================================================================
class PretrainArgs:
    def __init__(self):
        self.dataset = 'acm'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ACM 数据路径 (请修改为你存放 .mat 的实际路径)
        self.data_path = './DCRN/dataset/ACM3025.mat'
        
        # 预训练参数 (ACM 比较稳定，轮数可以适中)
        self.epochs_step1 = 30   # VGAE Recon
        self.epochs_step2 = 30   # Denoising
        self.epochs_step3 = 50   # Joint
        self.lr = 1e-3
        
        self.save_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        if not os.path.exists('./model_pretrain'):
            os.makedirs('./model_pretrain')

args = PretrainArgs()

# ... (辅助函数 contrastive_loss 保持不变，可以直接复制过来) ...
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    loss = -torch.mean(torch.log_softmax(sim_matrix, dim=1).diag())
    return loss

def main():
    # --- A. 数据加载 ---
    print(f">> Loading ACM data from {args.data_path}...")
    # 使用新写的 load_acm
    adj_sparse, feat, label, adj_label = load_acm(args.data_path, args.device)
    
    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    # ACM 特征维度较高 (1870)，中间层可以用 512 或 256
    gae_dims = [n_input, 512, 64] # 稍微加宽一点网络
    
    print(f">> Input Dim: {n_input}, Clusters: {n_clusters}")
    
    # --- B. 模型初始化 ---
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=n_input,
        hidden_dim=512, # 与 gae_dims 中间层对应
        num_clusters=n_clusters,
        gae_dims=gae_dims
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 重构 Loss 权重
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

    # =================================================================
    # Step 2: 预训练降噪视图
    # =================================================================
    print("\n=== Step 2: Pretraining Denoising View ===")
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

    # =================================================================
    # Step 3: 联合预训练
    # =================================================================
    print("\n=== Step 3: Joint Pretraining ===")
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

    # --- 保存 ---
    print(f">> Saving to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()