import numpy as np
import torch as th
import torch.nn as nn

class _Fusion(nn.Module):
    def __init__(self, cfg, input_sizes):
        """
        Base class for the fusion module

        :param cfg: Fusion config. See config.defaults.Fusion
        :param input_sizes: Input shapes
        """
        super().__init__()
        self.cfg = cfg
        self.input_sizes = input_sizes
        self.output_size = None

    def forward(self, inputs):
        raise NotImplementedError()

    @classmethod
    def get_weighted_sum_output_size(cls, input_sizes):
        flat_sizes = [np.prod(s) for s in input_sizes]
        assert all(s == flat_sizes[0] for s in flat_sizes), f"Fusion method {cls.__name__} requires the flat output" \
                                                            f" shape from all backbones to be identical." \
                                                            f" Got sizes: {input_sizes} -> {flat_sizes}."
        return [flat_sizes[0]]

    def get_weights(self, softmax=True):
        out = []
        if hasattr(self, "weights"):
            out = self.weights
            if softmax:
                out = nn.functional.softmax(self.weights, dim=-1)
        return out

    def update_weights(self, inputs, a):
        pass


class Mean(_Fusion):
    def __init__(self, cfg, input_sizes):
        """
        Mean fusion.

        :param cfg: Fusion config. See config.defaults.Fusion
        :param input_sizes: Input shapes
        """
        super().__init__(cfg, input_sizes)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def forward(self, inputs):
        return th.mean(th.stack(inputs, -1), dim=-1)


class _FusionBlock(nn.Module):
    """
    Fusion Block for Residual fusion.
    input -> (_, input_dim) -> norm -> (_, input_dim * expand) -> (_, input_dim) -> output
                   |                                                         |
                   -------------------------------+---------------------------
    """
    expand = 2

    def __init__(self, input_dim, act_func='relu', dropout=0., norm_eps=1e-5) -> None:
        super().__init__()
        latent_dim1 = input_dim * self.expand
        latent_dim2 = input_dim // self.expand
        if act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activate function must be ReLU or Tanh.')
        self.linear1 = nn.Linear(input_dim, latent_dim1, bias=False)
        self.linear2 = nn.Linear(latent_dim1, input_dim, bias=False)

        self.linear3 = nn.Linear(input_dim, latent_dim2, bias=False)
        self.linear4 = nn.Linear(latent_dim2, input_dim, bias=False)

        self.norm1 = nn.BatchNorm1d(input_dim, eps=norm_eps)
        self.norm2 = nn.BatchNorm1d(input_dim, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.block1(self.norm1(x))
        x = x + self.block2(self.norm2(x))
        return x

    def block1(self, x):
        return self.linear2(self.dropout1(self.act(self.linear1(x))))

    def block2(self, x):
        return self.linear4(self.dropout2(self.act(self.linear3(x))))

class adaptive_fusion(_Fusion):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
    """
    def __init__(self, cfg, input_sizes):
        super().__init__(cfg, input_sizes)
        self.weights = nn.Parameter(th.full((self.cfg.n_views,), 1 / self.cfg.n_views), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)
        self.map_layer = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.BatchNorm1d(768)
        block = _FusionBlock(768, act_func='relu')
        num_layers = 2
        self.fusion_modules = self._get_clones(block, num_layers)
        self.last_layer = nn.Sequential(
            nn.Linear(768, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )


    def forward(self, inputs):
        h = th.cat(inputs, dim=-1)
        z = self.map_layer(h)
        for mod in self.fusion_modules:
            z = mod(z)
        z = self.norm(z)
        z = self.last_layer(z)
        return z

    def _get_clones(self, module, N):
        """
        A deep copy will take a copy of the original object and will then recursively take a copy of the inner objects.
        The change in any of the models won’t affect the corresponding model.
        """
        import copy
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])




def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = nn.functional.softmax(weights, dim=0)
    out = th.sum(weights[None, None, :] * th.stack(tensors, dim=-1), dim=-1)
    return out


MODULES = {
    "mean": Mean,
    "adaptive_fusion": adaptive_fusion,
}


def get_fusion_module(cfg, input_sizes):
    return MODULES[cfg.method](cfg, input_sizes)
