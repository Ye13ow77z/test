import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

# === 0. 路径配置 ===
sys.path.append(os.getcwd()) 
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, accuracy_score
from scipy.optimize import linear_sum_assignment 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model_fusion import AdaDCRN_VGAE
from LUCE_CMC.src.lib.contrastive_loss import ClusterLoss

# ====================================================================
# 1. 专用数据加载器 (直接读取 ind.pubmed.* 文件)
#    (为了解决 utils_data.py 找不到文件的问题)
# ====================================================================
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_pubmed_raw(dataset_dir, device):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    dataset_str = 'pubmed'
    
    print(f">> Reading raw files from {dataset_dir}...")
    for i in range(len(names)):
        filename = os.path.join(dataset_dir, "ind.{}.{}".format(dataset_str, names[i]))
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing file: {filename}")
            
        with open(filename, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = objects
    test_idx_reorder = parse_index_file(os.path.join(dataset_dir, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    feat = torch.FloatTensor(np.array(features.todense())).to(device)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if labels.shape[1] > 1: 
        label = torch.LongTensor(np.argmax(labels, axis=1)).to(device)
    else:
        label = torch.LongTensor(labels).to(device)

    # 预处理邻接矩阵 D^-0.5(A+I)D^-0.5
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    
    # 转 Dense 用于 Loss (尝试)
    try:
        adj_dense = torch.FloatTensor(np.array(adj.todense())).to(device)
    except:
        print("!! Warning: Graph too large for Dense Tensor. Using None for adj_dense.")
        adj_dense = None

    indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col)).astype(np.int64))
    values = torch.from_numpy(adj_normalized.data.astype(np.float32))
    shape = torch.Size(adj_normalized.shape)
    adj_sparse = torch.sparse_coo_tensor(indices, values, shape).to(device)

    return adj_sparse, feat, label, adj_dense

# ====================================================================
# 2. 辅助函数 & 配置
# ====================================================================

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    loss = -torch.mean(torch.log_softmax(sim_matrix, dim=1).diag())
    return loss

def plot_tsne(z, labels, title, save_name):
    print(f">> Plotting t-SNE: {title}...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    z_2d = tsne.fit_transform(z)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f">> Saved to {save_name}")

class Args:
    def __init__(self):
        self.dataset = 'pubmed'
        self.pretrain_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        
        # === 核心配置 ===
        self.hidden_dim = 512 
        self.z_dim = 128     
        
        self.n_clusters = 0 
        self.n_input = 0    
        self.gae_dims = [] 
        self.use_cluster_proj = False 
        
        # 训练超参数 (PubMed 调优版)
        self.lr = 1e-4              # 降低学习率，防止破坏预训练特征
        self.epochs = 300
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_decay = 1e-4    # 降低正则
        
        # Loss 权重 (PubMed 专用调优)
        self.lambda_recon = 1.0
        self.lambda_kl_cluster = 1.0   # 大幅降低！防止聚类约束破坏特征
        self.lambda_vgae = 0.5         # 降低 VGAE KL 权重
        self.lambda_en_clu = 0.5       # 降低 En-CLU 权重
        self.cluster_temp = 0.5        # 提高温度，让分布更软
        self.l0_weight = 0             # 先关闭 L0
        self.lambda_contrastive = 0.1  # 降低对比损失
        
        self.warmup_epochs = 100       # 延长预热期

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
    weight = q**2 / (q.sum(0) + 1e-15)
    return (weight.t() / (weight.sum(1) + 1e-15)).t()

def vgae_kl_loss(mu, logstd):
    return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

def load_pretrained(model, path):
    if os.path.exists(path):
        print(f"Loading pretrained fusion model from {path}...")
        model.load_state_dict(torch.load(path, map_location=args.device), strict=False)
    else:
        print("!! Pretrain file not found. Please run pretrain_fusion_pubmed.py first !!")

# ====================================================================
# 主程序
# ====================================================================

if __name__ == "__main__":
    
    # 1. 数据加载
    print(f">> Loading {args.dataset} data...")
    adj_sparse, feat, label, adj_label = load_pubmed_raw(
        dataset_dir='./DCRN/dataset/pubmed/', 
        device=args.device
    )

    args.n_input = feat.shape[1]                
    args.n_clusters = len(torch.unique(label))
    args.gae_dims = [args.n_input, args.hidden_dim, args.z_dim]     
    
    print(f"\n>> Dataset: {args.dataset}")
    print(f">> Config: In={args.n_input}, Hidden={args.hidden_dim}, Z={args.z_dim}, Clusters={args.n_clusters}")

    # 2. 初始化模型
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=args.n_input,
        hidden_dim=args.hidden_dim,
        num_clusters=args.n_clusters,
        gae_dims=args.gae_dims,
        use_cluster_proj=args.use_cluster_proj
    ).to(args.device)

    # 3. 加载预训练权重
    load_pretrained(model, args.pretrain_path)
    
    # 4. 初始化 En-CLU Loss
    criterion_en_clu = ClusterLoss(args.n_clusters, args.cluster_temp, args.device).to(args.device)

    # 5. K-Means 初始化聚类中心
    print("Initializing cluster centers with K-Means...")
    model.eval() 
    with torch.no_grad():
        out_init = model(feat, adj_sparse)
        z_init = out_init['mu'].cpu().numpy()
        
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_init)
    
    model.head.weight.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    model.head.bias.data.fill_(0.0)
    
    acc, nmi, ari, f1 = eva(label.cpu().numpy(), y_pred)
    print(f"[Init] ACC: {acc:.4f} | NMI: {nmi:.4f}")

    # 6. 训练准备
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # 重构 Loss 权重
    # 如果 adj_label (dense) 存在，则计算权重；否则默认为 1
    if adj_label is not None:
        pos_sum = adj_label.sum().item()
        pos_weight_val = float((adj_label.shape[0]**2 - pos_sum) / (pos_sum + 1e-15))
        norm_val = adj_label.shape[0]**2 / float(((adj_label.shape[0]**2 - pos_sum) * 2) + 1e-15)
        pos_weight = torch.as_tensor(pos_weight_val, dtype=torch.float32, device=args.device)
    else:
        pos_weight = torch.tensor(1.0, device=args.device)
        norm_val = 1.0
    
    print("\n[Start Training]")
    model.train() 
    best_acc = 0
    best_epoch = 0
    best_model_path = f'best_fusion_model_{args.dataset}.pkl'
    
    for epoch in range(args.epochs):
        if epoch % 5 == 0: 
            with torch.no_grad():
                out = model(feat, adj_sparse)
                q_fused = out['q']
                p = target_distribution(q_fused)
        
        out = model(feat, adj_sparse)
        
        q_fused = out['q']
        q_gen   = out['q_gen']
        q_den   = out['q_den']
        
        # --- Loss Calculation ---
        # 1. DEC Loss
        kl_cluster_loss = F.kl_div(q_fused.log(), p, reduction='batchmean')
        
        # 2. En-CLU Loss
        loss_clu_gen = criterion_en_clu(q_gen, q_fused)
        loss_clu_den = criterion_en_clu(q_den, q_fused)
        en_clu_loss = loss_clu_gen + loss_clu_den
        
        # 3. Recon Loss (带安全检查)
        if adj_label is not None:
            recon_loss = norm_val * F.binary_cross_entropy_with_logits(
                out['adj_logits'].view(-1), adj_label.view(-1), pos_weight=pos_weight
            )
        else:
            recon_loss = torch.tensor(0.0, device=args.device)
        
        # 4. KL Divergence
        kl_vgae = vgae_kl_loss(out['mu'], out['logstd'])
        
        # 5. Contrastive Loss
        cl_loss = contrastive_loss(out['mu'], out['z_den'])
        
        # 6. L0 Loss
        l0_loss = out['l0_loss']
        curr_l0_w = args.l0_weight * min(1.0, epoch / 100.0)
        
        warmup = min(1.0, epoch / args.warmup_epochs)

        loss = args.lambda_kl_cluster * warmup * kl_cluster_loss \
               + args.lambda_recon * recon_loss \
               + args.lambda_vgae * kl_vgae \
               + args.lambda_en_clu * warmup * en_clu_loss \
               + args.lambda_contrastive * warmup * cl_loss \
               + curr_l0_w * l0_loss 

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if epoch % 10 == 0:
            y_pred = q_fused.argmax(1).cpu().numpy()
            acc, nmi, ari, f1 = eva(label.cpu().numpy(), y_pred)
            
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | En-CLU: {en_clu_loss.item():.4f} | ACC: {acc:.4f} | NMI: {nmi:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

    print("\nTraining Finished.")
    print(f"Best Epoch: {best_epoch} | Best ACC: {best_acc:.4f}")

    # === Final Evaluation ===
    if os.path.exists(best_model_path):
        print(f"\n>> Loading Best Model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        model.eval()
        with torch.no_grad():
            out = model(feat, adj_sparse)
            
            # 1. Network Output (Softmax)
            q_fused = out['q']
            y_pred_q = q_fused.argmax(1).cpu().numpy()
            q_acc, q_nmi, q_ari, q_f1 = eva(label.cpu().numpy(), y_pred_q)
            
            # 2. Post-KMeans on Fused Features
            z_fused = out['z_fused'].cpu().numpy()
            print(">> Running Post-Training K-Means on fused features...")
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=42)
            y_pred_k = kmeans.fit_predict(z_fused)
            best_acc_r, best_nmi_r, best_ari_r, _ = eva(label.cpu().numpy(), y_pred_k)

        print("="*60)
        print(f"FINAL RESULT on {args.dataset}:")
        print(f"1. Network Output (q):     ACC: {q_acc:.4f} | NMI: {q_nmi:.4f}")
        print(f"2. Post-KMeans (z_fused):  ACC: {best_acc_r:.4f} | NMI: {best_nmi_r:.4f}")
        print("="*60)
        
        # 自动画图
        try:
            plot_tsne(z_fused, label.cpu().numpy(), 
                      title=f"PubMed Fusion (ACC={q_acc:.4f})", 
                      save_name=f"tsne_pubmed_final.png")
        except Exception as e:
            print(f"Skip plotting: {e}")