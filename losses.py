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

def contrastive_loss(z1, z2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return -torch.mean(torch.sum(z1 * z2, dim=1))

def reconstruction_loss(x, x_recon):
    return F.mse_loss(x_recon, x)

def adagcl_reg_loss(denoise_weights):
    return torch.mean(denoise_weights)

# === 新增以下函数 ===
def vgae_kl_loss(mu, logstd):
    """
    计算 VGAE 的 KL 散度 Loss:
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + 2*logstd - mu^2 - exp(2*logstd))
    """
    return -0.5 / mu.size(0) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=1))