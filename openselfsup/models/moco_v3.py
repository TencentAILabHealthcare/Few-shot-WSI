import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
import torch.nn.functional as F

@MODELS.register_module
class MOCOv3(nn.Module):
    def __init__(self,
                 backbone,
                 projector=None,
                 predictor=None,
                 base_momentum=0.999,
                 temperature=1,
                 **kwargs):
        super(MOCOv3, self).__init__()

        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), 
            builder.build_neck(projector), 
            builder.build_neck(predictor))

        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), 
            builder.build_neck(projector))

        self.backbone = self.encoder_q[0]
        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update_key_encoder()
    

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        x1, x2 = img[:, 0, ...].contiguous(), img[:, 1, ...].contiguous()
        # compute query features
        q1, q2 = self.encoder_q(x1)[0], self.encoder_q(x2)[0]  # queries: NxC
        q1, q2 = F.normalize(q1), F.normalize(q2)
        
        with torch.no_grad():
            k1, k2 = self.encoder_k(x1)[0], self.encoder_k(x2)[0]
            k1, k2 = F.normalize(k1), F.normalize(k2)
            labels = torch.arange(len(k1)).cuda()
        
        logits1, logits2 = q1 @ k2.T, q2 @ k1.T
        loss = 2 * self.temperature \
            * (self.criterion(logits1/self.temperature, labels)
             + self.criterion(logits2/self.temperature, labels))

        return dict(loss=loss)


    def forward_test(self, img, **kwargs):
        backbone_feats = self.backbone(img)
        last_layer_feat = nn.functional.avg_pool2d(backbone_feats[-1],7)
        last_layer_feat = last_layer_feat.view(last_layer_feat.size(0), -1)
        return dict(backbone=last_layer_feat.cpu())

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_test(img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))


