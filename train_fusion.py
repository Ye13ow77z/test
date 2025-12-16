import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys

# === 0. 路径配置 ===
sys.path.append(os.getcwd()) 

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

# ====================================================================
# 智能权重加载 
# ====================================================================
def load_gae_to_vgae(model, pretrain_path, device):
    if pretrain_path is None or not os.path.exists(pretrain_path):
        print(f"!! Warning: Pretrain path {pretrain_path} not found. Using Random Init.")
        return

    print(f"Loading GAE weights from {pretrain_path}...")
    pretrained_dict = torch.load(pretrain_path, map_location=device)
    model_dict = model.state_dict()
    new_state_dict = {}
    
    matched_count = 0

    # === 手动映射表 ===
    # Key: 预训练文件中的键名 (GAE)
    # Value: 新模型中的键名列表 (AdaDCRN_VGAE)
    
    # 1. 第一层 (Input -> Hidden)
    # 预训练: gcn1.weight / bias
    # 新模型: 
    #   - view_gen.encoder_vgae.gcn_shared (生成视图编码器)
    #   - view_den.nblayers_0.0 (去噪视图的邻居变换)
    #   - view_den.selflayers_0.0 (去噪视图的自身变换)
    
    # 2. 第二层 (Hidden -> Z)
    # 预训练: gcn2.weight / bias
    # 新模型:
    #   - view_gen.encoder_vgae.encoder_mean.0 (均值映射的第一层 Linear)
    
def load_gae_to_vgae(model, pretrain_path, device):
    if pretrain_path is None or not os.path.exists(pretrain_path):
        print(f"!! Warning: Pretrain path {pretrain_path} not found. Using Random Init.")
        return

    print(f"Loading GAE weights from {pretrain_path}...")
    pretrained_dict = torch.load(pretrain_path, map_location=device)
    model_dict = model.state_dict()
    new_state_dict = {}
    
    matched_count = 0

    for k, v in pretrained_dict.items():
        # === 1. 匹配 Backbone (Input -> Hidden) ===
        # 预训练键名: gcn_shared.weight / bias
        if k.startswith('gcn_shared'):
            suffix = k.split('gcn_shared.')[-1] # weight 或 bias
            
            # (A) 赋值给生成视图的 Backbone
            target_gen = f"view_gen.encoder_vgae.gcn_shared.{suffix}"
            if target_gen in model_dict:
                new_state_dict[target_gen] = v
                matched_count += 1
            
            # (B) 赋值给去噪视图的 Backbone (初始化 DenoisingNet)
            # DenoisingNet 的结构是 nblayers_0 -> Sequential -> [0] Linear
            # 注意：AdaGCL 的 DenoisingNet 有 nblayers 和 selflayers 两组
            target_den_nb = f"view_den.nblayers_0.0.{suffix}"
            if target_den_nb in model_dict:
                new_state_dict[target_den_nb] = v
                matched_count += 1
                
            target_den_self = f"view_den.selflayers_0.0.{suffix}"
            if target_den_self in model_dict:
                new_state_dict[target_den_self] = v
                matched_count += 1

        # === 2. 匹配 MLP Head (Hidden -> Z) ===
        # 预训练键名: encoder_mean.0.weight, encoder_mean.2.bias 等
        elif k.startswith('encoder_mean'):
            # 直接提取 encoder_mean 之后的部分 (例如 "0.weight")
            suffix = k.split('encoder_mean.')[-1]
            
            # (A) 赋值给生成视图的 encoder_mean
            target_gen_mean = f"view_gen.encoder_vgae.encoder_mean.{suffix}"
            if target_gen_mean in model_dict:
                new_state_dict[target_gen_mean] = v
                matched_count += 1
            
            # (B) 可选：同时也赋值给 encoder_std (LogStd) 
            # 虽然 std 应该只有 mean 的一半大小或者随机，但用预训练的 mean 权重做初始化
            # 往往比随机初始化更稳定（相当于初始假设方差结构与均值结构相关）
            target_gen_std = f"view_gen.encoder_vgae.encoder_std.{suffix}"
            if target_gen_std in model_dict:
                new_state_dict[target_gen_std] = v
                matched_count += 1

    if matched_count == 0:
        print("!! 警告: 依然没有匹配到权重。请检查预训练文件是否是用 pretrain_gae_adagcl.py 生成的。")
        print("预训练文件键名示例:", list(pretrained_dict.keys()))
    else:
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"成功匹配并加载了 {matched_count} 个参数张量！(包括生成视图和去噪视图)")

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
    load_gae_to_vgae(model, args.pretrain_path, args.device)
    
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
    
    # Pos Weight 用于重构 Loss
    pos_weight_val = float(adj_label.shape[0]**2 - adj_label.sum()) / adj_label.sum()
    norm_val = adj_label.shape[0]**2 / float((adj_label.shape[0]**2 - adj_label.sum()) * 2)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=args.device)
    
    print("\n[Start Training]")
    model.train() # 切回训练模式

    for epoch in range(args.epochs):
        # Target Distribution 更新
        if epoch % 1 == 0:
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
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            y_pred = q_fused.argmax(1).cpu().numpy()
            acc, nmi, ari, f1 = eva(label.cpu().numpy(), y_pred)
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | En-CLU: {en_clu_loss.item():.4f} | ACC: {acc:.4f} | NMI: {nmi:.4f}")

    print("\nTraining Finished.")
    print(f"Final Result: ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | F1: {f1:.4f}")