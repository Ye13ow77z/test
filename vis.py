import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import sys

# 引入你的模型定义
from model_fusion import AdaDCRN_VGAE
from utils_data import load_graph_data

# ================= 配置 =================
dataset = 'cora'
model_path = 'best_fusion_model_cora.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ========================================

def plot_embedding(z, labels, title="t-SNE Visualization"):
    print(">> Running t-SNE...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    z_2d = tsne.fit_transform(z)
    
    print(">> Plotting...")
    plt.figure(figsize=(10, 8))
    
    # 获取唯一的类别
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        indices = labels == label
        plt.scatter(z_2d[indices, 0], z_2d[indices, 1], 
                    c=[color], 
                    label=f'Cluster {label}', 
                    s=10, alpha=0.6) # s是点的大小，alpha是透明度
    
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_name = f'{dataset}_tsne_acc78.png'
    plt.savefig(save_name, dpi=300)
    print(f">> Saved visualization to {save_name}")
    plt.show()

if __name__ == "__main__":
    # 1. 加载数据
    print(f"Loading {dataset} data...")
    adj, feat, label, adj_label = load_graph_data(dataset, path='./DCRN/dataset/', use_pca=False, device=device)
    adj_sparse = adj.to_sparse().to(device) # 记得转稀疏
    
    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    gae_dims = [n_input, 256, 50]
    
    # 2. 初始化模型
    print("Initializing Model...")
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=n_input,
        hidden_dim=256,
        num_clusters=n_clusters,
        gae_dims=gae_dims
    ).to(device)
    
    # 3. 加载权重
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: {model_path} not found!")
        sys.exit()
        
    # 4. 提取特征
    model.eval()
    with torch.no_grad():
        out = model(feat, adj_sparse)
        z_fused = out['z_fused'].cpu().numpy()
        # 如果你想看单纯 Gen 视图的效果，也可以取 out['mu'].cpu().numpy()
        
    # 5. 绘图
    plot_embedding(z_fused, label.cpu().numpy(), title=f"Attention Fusion on DBLP (ACC: 78.53%)")