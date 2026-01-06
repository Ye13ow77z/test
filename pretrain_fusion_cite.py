import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

sys.path.append(os.getcwd())

from model_fusion import AdaDCRN_VGAE
from utils_cite import load_citeseer # 引入新写的加载器

class PretrainArgs:
    def __init__(self):
        self.dataset = 'citeseer'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 指向你的数据目录 (根据截图)
        self.data_path = './DCRN/dataset/citeseer/' 
        
        self.epochs_step1 = 80
        self.epochs_step2 = 50
        self.epochs_step3 = 50
        self.lr = 1e-3
        
        self.save_path = f'./model_pretrain/{self.dataset}_fusion_pretrain.pkl'
        if not os.path.exists('./model_pretrain'):
            os.makedirs('./model_pretrain')

args = PretrainArgs()

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    loss = -torch.mean(torch.log_softmax(sim_matrix, dim=1).diag())
    return loss

def main():
    print(f">> Loading Citeseer data...")
    adj_sparse, feat, label, adj_label = load_citeseer(args.data_path, args.device)
    
    n_input = feat.shape[1]
    n_clusters = len(torch.unique(label))
    # Citeseer 特征维度~3703，隐藏层用 512
    gae_dims = [n_input, 512, 64] 
    
    print(f">> Input Dim: {n_input}, Clusters: {n_clusters}")
    
    model = AdaDCRN_VGAE(
        num_nodes=feat.shape[0],
        input_dim=n_input,
        hidden_dim=512,
        num_clusters=n_clusters,
        gae_dims=gae_dims
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Recon Loss Weight
    pos_weight_val = float(adj_label.shape[0]**2 - adj_label.sum()) / adj_label.sum()
    norm_val = adj_label.shape[0]**2 / float((adj_label.shape[0]**2 - adj_label.sum()) * 2)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=args.device)

    def compute_recon_loss(adj_logits):
        return norm_val * F.binary_cross_entropy_with_logits(
            adj_logits.view(-1), adj_label.view(-1), pos_weight=pos_weight
        )

    def compute_kl_loss(mu, logstd):
        return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))

    # Step 1
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

    # Step 2
    print("\n=== Step 2: Pretraining Denoising View ===")
    for epoch in range(args.epochs_step2):
        model.train()
        optimizer.zero_grad()
        out = model(feat, adj_sparse)
        loss = contrastive_loss(out['mu'], out['z_den']) + 1e-4 * out['l0_loss']
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Step 2 | Epoch {epoch} | Loss: {loss.item():.4f}")

    # Step 3
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

    print(f">> Saving to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()