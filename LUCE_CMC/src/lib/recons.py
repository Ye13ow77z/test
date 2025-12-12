import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ReConX1(nn.Module):
    def __init__(self, indim=256):
        super(ReConX1, self).__init__()
        self.indim = indim
        self.recon = nn.Sequential(
            nn.Linear(indim, 16*64, bias=False),
            nn.BatchNorm1d(num_features=16*64),
            nn.ReLU(inplace=True),
            # Lambda(lambda x: x.reshape(-1, 64, 4, 4)),

            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 40, bias=False))

    def forward(self, x):
        out = self.recon(x)
        return out

class ReConX2(nn.Module):
    def __init__(self, indim=256):
        super(ReConX2, self).__init__()
        self.indim = indim
        self.recon = nn.Sequential(
            nn.Linear(indim, 16*64, bias=False),
            nn.BatchNorm1d(num_features=16*64),
            nn.ReLU(inplace=True),
            # Lambda(lambda x: x.reshape(-1, 64, 4, 4)),

            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 254, bias=False))

    def forward(self, x):
        out = self.recon(x)
        return out

class ReConX3(nn.Module):
    def __init__(self, indim=256):
        super(ReConX3, self).__init__()
        self.indim = indim
        self.recon = nn.Sequential(
            nn.Linear(indim, 16*64, bias=False),
            nn.BatchNorm1d(num_features=16*64),
            nn.ReLU(inplace=True),
            # Lambda(lambda x: x.reshape(-1, 64, 4, 4)),

            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 928, bias=False))

    def forward(self, x):
        out = self.recon(x)
        return out