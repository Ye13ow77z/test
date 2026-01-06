import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys

# === 0. 路径配置 ===
sys.path.append(os.getcwd()) 
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, accuracy_score
from scipy.optimize import linear_sum_assignment 

# 1. 引入工具和模型
try:
    from utils_data import load_graph_data
    from utils_acm import load_acm
    from utils_cite import load_citeseer
except ImportError as e:
    print(f"Warning: Import failed: {e}. Make sure util files exist.")

from model_fusion import AdaDCRN_VGAE
from LUCE_CMC.src.lib.contrastive_loss import ClusterLoss

# ====================================================================
# 配置参数
# ====================================================================

class Args:
    def __init__(self):
        self.dataset = 'dblp'
        # 预训练权重路径
        self.pretrain_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        
        # 自动填充
        self.n_clusters = 0       
        self.n_input = 0       
        self.hidden_dim = 512
        self.gae_dims = [] 
        
        self.lr = 2e-4
        self.epochs = 400
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Loss 权重 
        self.lambda_recon = 1.0  
        self.lambda_kl_cluster = 10.0
        self.lambda_vgae = 1.0       
        self.lambda_en_clu = 10.0    
        self.cluster_temp = 0.5     

args = Args()

# ====================================================================
# 评估函数
# ====================================================================
def eva(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    remapping = {ind[0][i]: ind[1][i] for i in range(len(ind[0]))}
    y_pred_aligned = np.array([remapping.get(label, label) for label in y_pred])
    acc = accuracy_score(y_true, y_pred_aligned)
    f1 = f1_score(y_true, y_pred_aligned, average='macro')
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return acc, nmi, ari, f1

def target_distribution(q):
    # 数值稳定保护
    weight = q**2 / (q.sum(0) + 1e-15)
    return (weight.t() / (weight.sum(1) + 1e-15)).t()

def vgae_kl_loss(mu, logstd):
    return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

def load_pretrained(model, path):
    if os.path.exists(path):
        print(f"Loading full pretrained fusion model from {path}...")
        model.load_state_dict(torch.load(path, map_location=args.device), strict=False)
    else:
        print("Pretrain file not found, starting from scratch.")

# ====================================================================
# 主程序
# ====================================================================

if __name__ == "__main__":
    
    # === 1. 数据加载逻辑分支 ===
    if args.dataset == 'acm':
        print(f">> Loading ACM data via utils_acm...")
        # ACM 路径：直接指向 .mat 文件
        acm_path = './DCRN/dataset/ACM3025.mat'
        # load_acm 返回的已经是 sparse adj
        adj_sparse, feat, label, adj_label = load_acm(acm_path, args.device)
        
        # ACM 的隐藏层通常设置宽一点，如果你预训练用了 512，这里也得是 512
        # 假设你之前预训练用的 512 (根据之前的对话)
        args.hidden_dim = 512 
    elif args.dataset == 'citeseer':
        print(f">> Loading Citeseer data via utils_cite...")
        # Citeseer 路径：指向数据目录
        cite_path = './DCRN/dataset/citeseer/'
        adj_sparse, feat, label, adj_label = load_citeseer(cite_path, args.device)
        
        args.hidden_dim = 512 # Citeseer 通常用 512 隐藏层
    else:
        print(f">> Loading {args.dataset} data via utils_data...")
        adj, feat, label, adj_label = load_graph_data(
            args.dataset, 
            path='./DCRN/dataset/', 
            use_pca=False, 
            device=args.device
        )
        print("Converting Adjacency Matrix to Sparse Tensor...")
        adj_sparse = adj.to_sparse().to(args.device)
        args.hidden_dim = 512 # Cora/DBLP 默认

    # 动态更新参数
    args.n_input = feat.shape[1]                
    args.n_clusters = len(torch.unique(label))
    # 保持与预训练一致的维度结构
    args.gae_dims = [args.n_input, args.hidden_dim, 64 if args.hidden_dim == 512 else 50]     
    
    print(f"\n>> Dataset: {args.dataset}")
    print(f">> Input Dim: {args.n_input}, Clusters: {args.n_clusters}, Hidden: {args.hidden_dim}")

    # 2. 初始化模型
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=args.n_input,
        hidden_dim=args.hidden_dim,
        num_clusters=args.n_clusters,
        gae_dims=args.gae_dims
    ).to(args.device)

    # 3. 加载预训练权重
    load_pretrained(model, args.pretrain_path)
    
    # 4. 初始化 En-CLU Loss
    criterion_en_clu = ClusterLoss(args.n_clusters, args.cluster_temp, args.device).to(args.device)

    # 5. K-Means 初始化
    print("Initializing cluster centers with K-Means on Fused Features...")
    
    model.eval() 
    with torch.no_grad():
        out_init = model(feat, adj_sparse)
        z_init = out_init['mu'].cpu().numpy()
        
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_init)
    
    model.head.weight.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    model.head.bias.data.fill_(0.0)
    
    acc, nmi, ari, f1 = eva(label.cpu().numpy(), y_pred)
    print(f"[Init] ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

    # 6. 训练
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 如果用 CosineAnnealing，最小 LR 不宜为 0，防止后面完全不动
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Pos Weight 用于重构 Loss
    pos_weight_val = float(adj_label.shape[0]**2 - adj_label.sum()) / adj_label.sum()
    norm_val = adj_label.shape[0]**2 / float((adj_label.shape[0]**2 - adj_label.sum()) * 2)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=args.device)
    
    print("\n[Start Training]")
    model.train() 
    best_acc = 0
    best_epoch = 0
    best_model_path = f'best_fusion_model_{args.dataset}.pkl'
    
    for epoch in range(args.epochs):
        # Target Distribution 更新
        if epoch % 5 == 0: 
            with torch.no_grad():
                out = model(feat, adj_sparse)
                q_fused = out['q']
                p = target_distribution(q_fused)
        
        # Forward
        out = model(feat, adj_sparse)
        
        # 数值截断保护 (防止 NaN)
        q_fused = torch.clamp(out['q'], min=1e-15, max=1.0)
        q_gen   = torch.clamp(out['q_gen'], min=1e-15, max=1.0)
        q_den   = torch.clamp(out['q_den'], min=1e-15, max=1.0)
        
        adj_logits = out['adj_logits']
        mu = out['mu']
        logstd = out['logstd']
        l0_loss = out['l0_loss']
        
        # --- Loss ---
        # 1. DEC Loss (KL Divergence)
        kl_cluster_loss = F.kl_div(q_fused.log(), p, reduction='batchmean')
        
        # 2. En-CLU Loss (对比学习)
        loss_clu_gen = criterion_en_clu(q_gen, q_fused)
        loss_clu_den = criterion_en_clu(q_den, q_fused)
        en_clu_loss = loss_clu_gen + loss_clu_den
        
        # 3. Recon Loss (VGAE 重构)
        recon_loss = norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), adj_label.view(-1), pos_weight=pos_weight
        )
        
        # 4. VGAE KL
        kl_vgae = vgae_kl_loss(mu, logstd)
        
        # 总 Loss
        loss = args.lambda_kl_cluster * kl_cluster_loss \
             + args.lambda_recon * recon_loss \
             + args.lambda_vgae * kl_vgae \
             + args.lambda_en_clu * en_clu_loss \
             + 1e-3 * l0_loss 
        
        optimizer.zero_grad()
        loss.backward()
        
        # === 梯度裁剪 (防止爆炸) ===
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            y_pred = q_fused.argmax(1).cpu().numpy()
            acc, nmi, ari, f1 = eva(label.cpu().numpy(), y_pred)
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | En-CLU: {en_clu_loss.item():.4f} | ACC: {acc:.4f} | NMI: {nmi:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"   >> [New Best] Model saved (ACC: {best_acc:.4f})")
            
            # 打印注意力权重看看 (如果有返回)
            if 'att_weights' in out:
                att_w = out['att_weights'].mean(dim=0).squeeze().detach().cpu().numpy()
                print(f"   >> Attn: Gen {att_w[0]:.3f} | Den {att_w[1]:.3f}")

    print("\nTraining Finished.")
    print(f"Recorded Best Epoch: {best_epoch} | Best ACC: {best_acc:.4f}")

    # === 加载最佳模型并重新评估 ===
    if os.path.exists(best_model_path):
        print(f"\n>> Loading Best Model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        model.eval()
        with torch.no_grad():
            out = model(feat, adj_sparse)
            q_fused = out['q']
            y_pred = q_fused.argmax(1).cpu().numpy()
            final_acc, final_nmi, final_ari, final_f1 = eva(label.cpu().numpy(), y_pred)
            
        print("="*60)
        print(f"FINAL BEST RESULT on {args.dataset}:")
        print(f"ACC: {final_acc:.4f} | NMI: {final_nmi:.4f} | ARI: {final_ari:.4f} | F1: {final_f1:.4f}")
        print("="*60)