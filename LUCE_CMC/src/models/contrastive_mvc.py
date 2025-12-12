import torch as th
import torch.nn as nn

import helpers
from lib.loss import Loss
from lib.optimizer import Optimizer
from lib.backbones import Backbones, MLP
from lib.fusion import get_fusion_module
from models.clustering_module import DDC
from models.model_base import ModelBase
from models.token_transformer import Token_transformer
from torch.nn.functional import normalize


class LUCECMC(ModelBase):
    def __init__(self, cfg):
        """
        Implementation of the CoMVC model.

        :param cfg: Model config. See `config.defaults.CoMVC` for documentation on the config object.
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = self.projections = None
        self.backbones = Backbones(cfg.backbone_configs)
        self.fusion = get_fusion_module(cfg.fusion_config, self.backbones.output_sizes)
        bb_sizes = self.backbones.output_sizes
        assert all([bb_sizes[0] == s for s in bb_sizes]), f"CoMVC requires all backbones to have the same " \
                                                          f"output size. Got: {bb_sizes}"
        if cfg.projector_config is None:
            self.projector = nn.Identity()
        else:
            self.projector = MLP(cfg.projector_config, input_size=bb_sizes[0])
        # Define clustering module
        self.ddc = DDC(input_dim=256, cfg=cfg.cm_config)
        self.ddc_output = DDC(input_dim=512, cfg=cfg.cm_config)
        # Define loss-module
        self.loss = Loss(cfg=cfg.loss_config)
        # Initialize weights.
        self.apply(helpers.he_init_weights)
        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())
        self.trans = Token_transformer(dim=256)

        self.instance_projector = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 20),
            nn.Softmax(dim=1)
        )



    def forward(self, views):
        self.backbone_outputs = self.backbones(views)
        self.view1 = views[0]
        self.view2 = views[1]
        self.view3 = views[2]
        #对每个视角进行Trans
        a1 = self.backbone_outputs[0].unsqueeze(1)
        a2 = self.backbone_outputs[1].unsqueeze(1)
        a3 = self.backbone_outputs[2].unsqueeze(1)
        b1 = self.trans(a1)
        b2 = self.trans(a2)
        b3 = self.trans(a3)
        b1 = b1.transpose(0, 1)
        b2 = b2.transpose(0, 1)
        b3 = b3.transpose(0, 1)
        b1 = b1.mean(dim=0)
        b2 = b2.mean(dim=0)
        b3 = b3.mean(dim=0)
        self.backbone_outputs[0] = b1
        self.backbone_outputs[1] = b2
        self.backbone_outputs[2] = b3
        self.z1 = normalize(self.instance_projector(self.backbone_outputs[0]), dim=1)
        self.z2 = normalize(self.instance_projector(self.backbone_outputs[1]), dim=1)
        self.z3 = normalize(self.instance_projector(self.backbone_outputs[2]), dim=1)
        self.fused = self.fusion(self.backbone_outputs)
        self.projections = self.projector(th.cat(self.backbone_outputs, dim=0))
        self.backbone_ddc01, self.backbone_hidden01 = self.ddc(self.z1)
        self.backbone_ddc02, self.backbone_hidden02 = self.ddc(self.z2)
        self.backbone_ddc03, self.backbone_hidden03 = self.ddc(self.z3)
        self.output, self.hidden = self.ddc(self.fused)
        return self.output

