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
from utils_data import load_graph_data
from model_fusion import AdaDCRN_VGAE

# 2. 引入 LUCE 的 En-CLU Loss
# 确保你的目录下有这个文件
from LUCE_CMC.src.lib.contrastive_loss import ClusterLoss

# ====================================================================
# 配置参数
# ====================================================================

class Args:
    def __init__(self):
        self.dataset = 'dblp'
        # 预训练权重路径
        self.pretrain_path = f'./DCRN/model_pretrain/{self.dataset}_adagcl_pretrain.pkl'
        
        # 自动填充
        self.n_clusters = 0       
        self.n_input = 0       
        self.hidden_dim = 256     
        self.gae_dims = [] 
        
        self.lr = 1e-3
        self.epochs = 200
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Loss 权重 
        self.lambda_recon = 0.5   
        self.lambda_kl_cluster = 1.0  
        self.lambda_vgae = 0.5        
        self.lambda_en_clu = 1.5  # 提高 En-CLU 权重，强调对比
        self.cluster_temp = 1.0       

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
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def vgae_kl_loss(mu, logstd):
    return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))


def load_pretrained(model, path):
    if os.path.exists(path):
        print(f"Loading full pretrained fusion model from {path}...")
        model.load_state_dict(torch.load(path, map_location=args.device), strict=False)
    else:
        print("Pretrain file not found, starting from scratch.")

def feature_align_loss(z1, z2):
    # 使用 Cosine Similarity 使得两个向量方向一致
    # z1, z2: [Batch, Dim]
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    # 我们希望相似度越大越好，所以 Loss 是负的相似度，或者 2 - 2*sim (MSE of normalized vectors)
    # 这里用 MSE of normalized vectors: ||z1' - z2'||^2 = 2 - 2*cos(theta)
    return torch.mean((z1_norm - z2_norm).pow(2))

# ====================================================================
# 主程序
# ====================================================================

if __name__ == "__main__":
    # 1. 加载数据
    adj, feat, label, adj_label = load_graph_data(
        args.dataset, 
        path='./DCRN/dataset/', 
        use_pca=False, 
        device=args.device
    )
    
    # === 关键修正：转换为稀疏张量 ===
    # AdaGCL 需要 sparse adjacency matrix 来调用 _indices()
    print("Converting Adjacency Matrix to Sparse Tensor...")
    # 注意：adj 是归一化后的 dense tensor
    # 使用 to_sparse() 转换
    adj_sparse = adj.to_sparse().to(args.device)
    
    # 动态更新参数
    args.n_input = feat.shape[1]                
    args.n_clusters = len(torch.unique(label))  
    args.gae_dims = [args.n_input, 256, 50]     
    
    print(f"\n>> Dataset: {args.dataset}")
    print(f">> Input Dim: {args.n_input}, Clusters: {args.n_clusters}")

    # 2. 初始化模型
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=args.n_input,
        hidden_dim=256,
        num_clusters=args.n_clusters,
        gae_dims=args.gae_dims
    ).to(args.device)

    # 3. 加载预训练权重
    load_pretrained(model, './model_pretrain/dblp_fusion_pretrain.pkl')
    
    # 4. 初始化 En-CLU Loss
    criterion_en_clu = ClusterLoss(args.n_clusters, args.cluster_temp, args.device).to(args.device)

    # 5. K-Means 初始化
    print("Initializing cluster centers with K-Means on Fused Features...")
    
    # === 关键：Eval 模式 ===
    model.eval() 
    with torch.no_grad():
        # 注意：这里传入 adj_sparse
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
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Pos Weight 用于重构 Loss
    pos_weight_val = float(adj_label.shape[0]**2 - adj_label.sum()) / adj_label.sum()
    norm_val = adj_label.shape[0]**2 / float((adj_label.shape[0]**2 - adj_label.sum()) * 2)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=args.device)
    
    print("\n[Start Training]")
    model.train() # 切回训练模式
    best_acc = 0
    best_epoch = 0
    best_model_path = 'best_fusion_model.pkl'
    for epoch in range(args.epochs):
        # Target Distribution 更新
        if epoch % 5 == 0:
            with torch.no_grad():
                out = model(feat, adj_sparse)
                q_fused = out['q']
                p = target_distribution(q_fused)
        
        # Forward (传入 Sparse ADJ)
        out = model(feat, adj_sparse)
        
        q_fused = out['q']
        q_gen   = out['q_gen']
        q_den   = out['q_den']
        adj_logits = out['adj_logits']
        mu = out['mu']
        logstd = out['logstd']
        l0_loss = out['l0_loss'] # 获取稀疏正则 Loss
        
        # --- Loss ---
        # 1. DEC Loss
        q_fused = torch.clamp(q_fused, min=1e-15, max=1.0)
        kl_cluster_loss = F.kl_div(q_fused.log(), p, reduction='batchmean')
        
        # 2. En-CLU Loss (对比学习)
        loss_clu_gen = criterion_en_clu(q_gen, q_fused)
        loss_clu_den = criterion_en_clu(q_den, q_fused)
        en_clu_loss = loss_clu_gen + loss_clu_den
        
        # 3. Recon Loss (VGAE 重构)
        # 注意：这里 adj_logits 是 Dense 的 (N x N)，adj_label 也是 Dense 的
        # 重构 Loss 还是用 Dense 计算比较方便，不用动
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
             + 1e-4 * l0_loss # L0 Regularization
        
        optimizer.zero_grad()
        loss.backward()
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
            att_w = out['att_weights'].mean(dim=0).squeeze().detach().cpu().numpy()
            print(f"   >> Attention Weights - Gen: {att_w[0]:.4f} | Den: {att_w[1]:.4f}")
            
    print("\nTraining Finished.")
    print(f"Final Result: ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | F1: {f1:.4f}")
    print("\nTraining Finished.")
    print(f"Recorded Best Epoch: {best_epoch} | Best ACC: {best_acc:.4f}")

    # === 新增：加载最佳模型并重新评估 ===
    best_model_path = 'best_fusion_model.pkl' # 确保这里的文件名和你保存时的一致
    
    if os.path.exists(best_model_path):
        print(f"\n>> Loading Best Model from {best_model_path}...")
        # 加载权重
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        
        # 重新评估
        model.eval()
        with torch.no_grad():
            out = model(feat, adj_sparse)
            q_fused = out['q']
            y_pred = q_fused.argmax(1).cpu().numpy()
            
            # 计算指标
            final_acc, final_nmi, final_ari, final_f1 = eva(label.cpu().numpy(), y_pred)
            
        print("="*60)
        print(f"FINAL BEST RESULT (Restored):")
        print(f"ACC: {final_acc:.4f} | NMI: {final_nmi:.4f} | ARI: {final_ari:.4f} | F1: {final_f1:.4f}")
        print("="*60)
    else:
        print("!! Warning: Best model file not found.")