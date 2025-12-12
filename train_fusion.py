import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, accuracy_score
from scipy.optimize import linear_sum_assignment 

# 1. 引入工具和模型
from utils_data import load_graph_data
from model_fusion import AdaDCRN_VGAE

# ====================================================================
# 配置参数
# ====================================================================

class Args:
    def __init__(self):
        self.dataset = 'cora'
        self.n_clusters = 7       
        self.n_input = 1433       
        self.hidden_dim = 256     
        self.gae_dims = [1433, 256, 50] 
        
        self.lr = 1e-3
        self.epochs = 200
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 预训练权重 (GAE的权重可以部分复用于 VGAE)
        self.pretrain_path = './DCRN/model_pretrain/cora_gae_pretrain.pkl'
        
        # Loss 权重
        self.lambda_recon = 1.0   
        self.lambda_kl_cluster = 1.0  # 聚类 KL
        self.lambda_vgae = 0.5        # VGAE 正则化 KL (通常设小一点，防止 Latent Space 过度平滑)

args = Args()

# ====================================================================
# 评估函数
# ====================================================================

def eva(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
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
    """
    计算 VGAE 的 KL 散度 Loss:
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + 2*logstd - mu^2 - exp(2*logstd))
    """
    return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

# ====================================================================
# 智能权重加载 (适配 VGAE)
# ====================================================================

def load_gae_to_vgae(model, pretrain_path, device):
    if pretrain_path is None or not os.path.exists(pretrain_path):
        print(f"!! Info: Pretrain path is None or not found. Random Init.")
        return

    print(f"Loading GAE weights from {pretrain_path} into VGAE...")
    pretrained_dict = torch.load(pretrain_path, map_location=device)
    model_dict = model.state_dict()
    new_state_dict = {}
    
    # 映射逻辑：
    # GAE: gcn1 -> VGAE: layers.0
    # GAE: gcn2 -> VGAE: gc_mu (我们把训练好的特征映射给均值网络)
    # VGAE: gc_logstd (这个层没有预训练，保持随机初始化)
    
    layer_map_shared = {'gcn1': '0'} # 第一层是共享的
    layer_map_head = {'gcn2': 'gc_mu'} # 第二层给 mu

    for key, value in pretrained_dict.items():
        parts = key.split('.')
        prefix = parts[0]
        suffix = '.'.join(parts[1:]) 
        
        # 1. 处理共享层 (gcn1)
        if prefix in layer_map_shared:
            idx = layer_map_shared[prefix]
            # Generative View
            gen_key = f'view_gen.layers.{idx}.{suffix}'
            new_state_dict[gen_key] = value
            # Denoising View
            den_key = f'view_den.layers.{idx}.{suffix}'
            new_state_dict[den_key] = value
            
        # 2. 处理 Head 层 (gcn2 -> gc_mu)
        if prefix in layer_map_head:
            target_layer = layer_map_head[prefix]
            # Generative View (给 mu)
            gen_key = f'view_gen.{target_layer}.{suffix}'
            new_state_dict[gen_key] = value
            # Denoising View (DenoisingView 还是普通的 GCN，所以还是 layers.1)
            # 注意：Denoising View 结构没变，所以 gcn2 对应 layers.1
            den_key = f'view_den.layers.1.{suffix}'
            new_state_dict[den_key] = value

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)
    print("Pre-trained weights loaded! (gc_logstd initialized randomly)")

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
    num_nodes = feat.shape[0]

    # 2. 初始化模型 (VGAE版本)
    model = AdaDCRN_VGAE(
        num_nodes=num_nodes,
        input_dim=args.n_input,
        hidden_dim=256,
        num_clusters=args.n_clusters,
        gae_dims=args.gae_dims
    ).to(args.device)

    # 3. 加载权重
    load_gae_to_vgae(model, args.pretrain_path, args.device)

    # 4. K-Means 初始化
    print("Initializing cluster centers with K-Means...")
    with torch.no_grad():
        # VGAE 返回多个值，我们只需要 mu 作为特征来聚类
        # 因为 z 是采样的，有随机性，用 mu 初始化更稳定
        _, _, mu, _ = model.view_gen(feat, adj)
        z_np = mu.cpu().numpy()
        
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_np)
    
    model.head.weight.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    model.head.bias.data.fill_(0.0)
    
    acc, nmi, ari, f1 = eva(label.cpu().numpy(), y_pred)
    print(f"[Init] ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | F1: {f1:.4f}")

    # 5. 训练
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    pos_weight_val = float(adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) / adj_label.sum()
    norm_val = adj_label.shape[0] * adj_label.shape[0] / float((adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) * 2)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=args.device)
    
    print("\n[Start Training VGAE]")
    for epoch in range(args.epochs):
        model.train()
        
        if epoch % 1 == 0:
            with torch.no_grad():
                out = model(feat, adj)
                q = out['q']
                p = target_distribution(q)
        
        # Forward
        out = model(feat, adj)
        q = out['q']
        adj_logits = out['adj_logits']
        mu = out['mu']
        logstd = out['logstd']
        
        # 1. 聚类 Loss
        q = torch.clamp(q, min=1e-15, max=1.0)
        kl_cluster_loss = F.kl_div(q.log(), p, reduction='batchmean')
        
        # 2. 重构 Loss
        recon_loss = norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), 
            adj_label.view(-1), 
            pos_weight=pos_weight
        )
        
        # 3. VGAE KL Loss (Regularization)
        kl_vgae = vgae_kl_loss(mu, logstd)
        
        # 总 Loss
        loss = args.lambda_kl_cluster * kl_cluster_loss \
             + args.lambda_recon * recon_loss \
             + args.lambda_vgae * kl_vgae
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            y_pred = q.argmax(1).cpu().numpy()
            acc, nmi, ari, f1 = eva(label.cpu().numpy(), y_pred)
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL_VGAE: {kl_vgae.item():.4f} | ACC: {acc:.4f} | NMI: {nmi:.4f} | F1: {f1:.4f}")

    print("\nTraining Finished.")
    print(f"Final Result: ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | F1: {f1:.4f}")