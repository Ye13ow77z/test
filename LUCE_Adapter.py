import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    轻量级注意力融合层：输入 K 个视图，输出加权后的全局视图。
    """
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        # 一个简单的 MLP 来计算注意力分数
        self.att = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False) # 输出标量分数
        )
    
    def forward(self, z_list):
        # z_list: [z1, z2, z3] -> 每个都是 [Batch, Dim]
        # 堆叠 -> [Batch, 3, Dim]
        h = torch.stack(z_list, dim=1) 
        
        # 计算每个视图的分数 -> [Batch, 3, 1]
        att_score = self.att(h)
        
        # Softmax 归一化，保证权重之和为 1
        weights = F.softmax(att_score, dim=1) 
        
        # 加权求和: (Batch, 3, Dim) * (Batch, 3, 1) -> Sum dim 1 -> (Batch, Dim)
        z_global = torch.sum(h * weights, dim=1)
        
        return z_global, weights

class DDC(nn.Module):
    """ DDC 聚类头 """
    def __init__(self, input_dim, n_clusters):
        super(DDC, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_clusters)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.mlp(x)
        return self.softmax(logits) # 输出概率分布