import torch
import torch.nn.functional as F

def target_distribution(q):
    weight = (q ** 2) / (q.sum(0) + 1e-6)
    return (weight.t() / (weight.sum(1) + 1e-6)).t()

def kl_loss(q, p):
    return F.kl_div((q + 1e-8).log(), p, reduction='batchmean')

def luce_clustering_loss(q_gen, q_den, q_fused, p):
    loss_gen = F.kl_div((q_gen + 1e-8).log(), p, reduction='batchmean')
    loss_den = F.kl_div((q_den + 1e-8).log(), p, reduction='batchmean')
    loss_fused = F.kl_div((q_fused + 1e-8).log(), p, reduction='batchmean')
    return loss_gen + loss_den + loss_fused

def contrastive_loss(z1, z2, temperature=0.5):
    """
    改进的对比损失：使用 NT-Xent（对称化的对比损失）
    
    目标：最大化同一实例的两个视图的相似度，最小化不同实例的相似度
    """
    z1 = F.normalize(z1, dim=1)  # [N, z_dim]
    z2 = F.normalize(z2, dim=1)  # [N, z_dim]
    
    # 计算相似度矩阵
    sim_matrix = torch.mm(z1, z2.t()) / temperature  # [N, N]
    
    # 对角线元素是正样本对
    pos_sim = torch.diag(sim_matrix)  # [N]
    
    # 使用 logsumexp 进行数值稳定的计算
    # loss = -log(exp(pos_sim) / sum(exp(sim)))
    loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
    
    return loss.mean()

def reconstruction_loss(x, x_recon):
    return F.mse_loss(x_recon, x)

def adagcl_reg_loss(denoise_weights):
    return torch.mean(denoise_weights)

def vgae_kl_loss(mu, logstd):
    """
    计算 VGAE 的 KL 散度 Loss:
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + 2*logstd - mu^2 - exp(2*logstd))
    """
    return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))


def triplet_loss(anchor, positive, negative, margin=1.0):
    """三元组损失：增强聚类间的分离性"""
    pos_dist = (anchor - positive).pow(2).sum(dim=1)
    neg_dist = (anchor - negative).pow(2).sum(dim=1)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def cluster_separation_loss(z, labels, margin=1.0):
    """
    聚类分离损失：增大不同类别之间的距离
    """
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=z.device)
    
    centers = []
    for lbl in unique_labels:
        mask = labels == lbl
        centers.append(z[mask].mean(dim=0))
    centers = torch.stack(centers)  # [K, D]
    
    # 计算聚类中心间的距离
    dist_matrix = torch.cdist(centers, centers)
    
    # 取上三角（不包括对角线）
    mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
    inter_cluster_dist = dist_matrix[mask]
    
    # 希望聚类间距离大于margin
    loss = F.relu(margin - inter_cluster_dist).mean()
    return loss