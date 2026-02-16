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

# 路径配置
sys.path.append(os.getcwd())

from model_fusion import AdaDCRN_VGAE

# ====================================================================
# 0. 专用数据加载器 (直接读取 ind.pubmed.* 文件)
#    是为了绕过 utils_data.py 的路径报错问题
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
            raise FileNotFoundError(f"Missing file: {filename}. Please run download.py first.")
            
        with open(filename, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = objects
    test_idx_reorder = parse_index_file(os.path.join(dataset_dir, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    # 1. 构建特征矩阵
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    feat = torch.FloatTensor(np.array(features.todense())).to(device)

    # 2. 构建图结构
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    # 3. 处理标签
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if labels.shape[1] > 1: 
        label = torch.LongTensor(np.argmax(labels, axis=1)).to(device)
    else:
        label = torch.LongTensor(labels).to(device)

    # 4. 预处理邻接矩阵 (Standard GCN Normalization)
    # 这里不做额外的自环检查，只做标准的 D^-0.5(A+I)D^-0.5
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    
    # Loss 计算用的 Dense Adj (尽量做近似以节省显存)
    # PubMed 比较大，如果显存爆了，可以将 use_dense_loss 设为 False
    use_dense_loss = True
    adj_dense = None
    if use_dense_loss:
        try:
            # 尝试转 dense，如果不行就捕获异常
            # 注意：这里我们用未归一化的原始 adj 做 loss target 也可以，或者归一化后的
            # 通常 DCRN 使用原始 A+I 或者 A 做重构目标
            adj_dense = torch.FloatTensor(np.array(adj.todense())).to(device)
        except:
            print("!! Warning: Graph too large for Dense Tensor. Will use weighted sampling or skip.")
            use_dense_loss = False

    indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col)).astype(np.int64))
    values = torch.from_numpy(adj_normalized.data.astype(np.float32))
    shape = torch.Size(adj_normalized.shape)
    adj_sparse = torch.sparse_coo_tensor(indices, values, shape).to(device)

    print(f">> Data Loaded: Nodes={feat.shape[0]}, Dim={feat.shape[1]}, Classes={len(torch.unique(label))}")
    return adj_sparse, feat, label, adj_dense

# ====================================================================
# 1. 辅助 Loss 函数
# ====================================================================
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    loss = -torch.mean(torch.log_softmax(sim_matrix, dim=1).diag())
    return loss

# ====================================================================
# 2. 配置参数
# ====================================================================
class PretrainArgs:
    def __init__(self):
        self.dataset = 'pubmed'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 按照 ACM 的逻辑配置轮数
        # PubMed 较大，如果想跑快点可以减少，但为了收敛建议保持
        self.epochs_step1 = 120 
        self.epochs_step2 = 120 
        self.epochs_step3 = 120 
        self.lr = 1e-3
        
        self.save_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        if not os.path.exists('./model_pretrain'):
            os.makedirs('./model_pretrain')

        self.hidden_dim = 512
        self.z_dim = 128

args = PretrainArgs()

def main():
    # --- A. 数据加载 ---
    adj_sparse, feat, label, adj_label = load_pubmed_raw(
        dataset_dir='./DCRN/dataset/pubmed/', 
        device=args.device
    )
    
    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    num_nodes = feat.shape[0]
    gae_dims = [n_input, args.hidden_dim, args.z_dim]
    
    print(f">> Pretrain Config: Input={n_input}, Hidden={args.hidden_dim}, Z={args.z_dim}, Clusters={n_clusters}")

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
    
    # Loss 权重准备
    if adj_label is not None:
        pos_sum = adj_label.sum()
        n_nodes_sq = adj_label.shape[0]**2
        pos_weight_val = float(n_nodes_sq - pos_sum) / (pos_sum + 1e-15)
        norm_val = n_nodes_sq / float((n_nodes_sq - pos_sum) * 2 + 1e-15)
        pos_weight = torch.as_tensor(pos_weight_val, dtype=torch.float32, device=args.device)
    else:
        # Fallback if dense adj is too big
        pos_weight = torch.tensor(1.0, device=args.device)
        norm_val = 1.0

    def compute_recon_loss(adj_logits):
        if adj_label is None: return torch.tensor(0.0, device=args.device)
        return norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), adj_label.view(-1), pos_weight=pos_weight
        )

    def compute_kl_loss(mu, logstd):
        return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

    # ==========================================
    # Step 1: VGAE 生成视图预热 (纯结构重构)
    # ==========================================
    print("\n=== Step 1: Pretraining Generative View (Reconstruction) ===")
    for epoch in range(args.epochs_step1):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        # 仅使用结构重构 + KL，不加特征 Loss
        loss = compute_recon_loss(out['adj_logits']) + compute_kl_loss(out['mu'], out['logstd'])
        
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 1 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # ==========================================
    # Step 2: Denoising View Pretraining
    # ==========================================
    print("\n=== Step 2: Pretraining Denoising View (Contrastive) ===")
    for epoch in range(args.epochs_step2):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        z_gen = out['mu'].detach() 
        z_den = out['z_den'] 
        
        loss = contrastive_loss(z_gen, z_den) + 1e-4 * out['l0_loss']
        
        # 按照 ACM 逻辑，这里只保留极小的辅助特征 loss (0.1) 或者完全去掉
        # 这里保留 0.1 以防模型坍塌，但不使用强约束
        if 'recon_den' in out:
             loss += 0.1 * F.mse_loss(out['recon_den'], feat)

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 2 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # ==========================================
    # Step 3: Joint Training (联合微调)
    # ==========================================
    print("\n=== Step 3: Joint Pretraining (All Components) ===")
    for epoch in range(args.epochs_step3):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        
        loss_recon = compute_recon_loss(out['adj_logits'])
        loss_kl = compute_kl_loss(out['mu'], out['logstd'])
        loss_cl = contrastive_loss(out['mu'], out['z_den'])
        loss_l0 = 1e-4 * out['l0_loss']
        
        # 纯 ACM 逻辑：联合训练通常不加特征重构 loss，或者权重很小
        # 这里我们只使用上述四部分
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