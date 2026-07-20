import torch
import torch.nn as nn
import torch.nn.functional as F


class CCGF(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, features):
        """
        Args:
            inputs: (tensor) [N, C] logits
            targets: (tensor) [N] (0 <= targets < C)
            features:(tensor) [N,128] 
        """

        mask_0 = (targets == 0)
        mask_1 = (targets == 1)
        features_0 = features[mask_0]
        features_1 = features[mask_1]

        c0 = features_0.mean(dim=0)
        c1 = features_1.mean(dim=0)


        centers = torch.zeros_like(features)
        centers[mask_0] = c0
        centers[mask_1] = c1

        dot_product = (features * centers).sum(dim=1)
        feat_norm = features.norm(dim=1, p=2) + 1e-8
        center_norm = centers.norm(dim=1, p=2) + 1e-8
        cos_sim = dot_product / (feat_norm * center_norm)

        D = (cos_sim + 1) * 0.5
        d_min = D.min()
        d_max = D.max()
        norm_D = (D - d_min) / (d_max - d_min + 1e-8)
      
        gamma_i = self.gamma * ( 2 - norm_D )

        logpt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(dim=-1, index=targets.unsqueeze(1))
        pt = pt.gather(dim=-1, index=targets.unsqueeze(1))

        loss = - (1 - pt) **  gamma_i.unsqueeze(1)  * logpt

        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)

            at = self.alpha.gather(0, targets)
            loss = at.unsqueeze(0) * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
