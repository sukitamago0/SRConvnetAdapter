import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """ESRGAN/Real-ESRGAN-style VGG feature extractor."""

    LAYER_MAP = {
        'conv1_2': 3,
        'conv2_2': 8,
        'conv3_4': 17,
        'conv4_4': 26,
        'conv5_4': 35,
    }

    def __init__(self, layer_name_list=None, use_input_norm=True, range_norm=False):
        super().__init__()
        if layer_name_list is None:
            layer_name_list = ['conv5_4']
        self.layer_name_list = list(layer_name_list)
        self.use_input_norm = bool(use_input_norm)
        self.range_norm = bool(range_norm)

        vgg_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        max_idx = max(self.LAYER_MAP[name] for name in self.layer_name_list)
        self.vgg_net = nn.Sequential(*[vgg_features[i] for i in range(max_idx + 1)])
        self.vgg_net.eval()
        for p in self.vgg_net.parameters():
            p.requires_grad_(False)

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

    def forward(self, x: torch.Tensor):
        if self.range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for idx, layer in enumerate(self.vgg_net):
            x = layer(x)
            for name in self.layer_name_list:
                if idx == self.LAYER_MAP[name]:
                    output[name] = x.clone()
        return output


class PerceptualLoss(nn.Module):
    """Real-ESRGAN/ESRGAN-style perceptual loss.

    Main mode is VGG pre-activation features. LPIPS is kept as optional ablation.
    """

    def __init__(self, percep_type: str = 'vgg_preact', layer_weights=None, loss_weight: float = 1.0):
        super().__init__()
        self.percep_type = str(percep_type)
        self.loss_weight = float(loss_weight)
        if layer_weights is None:
            layer_weights = {'conv5_4': 1.0}
        self.layer_weights = dict(layer_weights)

        if self.percep_type == 'vgg_preact':
            self.vgg = VGGFeatureExtractor(layer_name_list=list(self.layer_weights.keys()), use_input_norm=True, range_norm=False)
            self.lpips_fn = None
        elif self.percep_type == 'lpips':
            import lpips
            self.lpips_fn = lpips.LPIPS(net='vgg').eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
            self.vgg = None
        else:
            raise ValueError(f'Unsupported percep_type: {self.percep_type}')

    def forward(self, pred_m11: torch.Tensor, target_m11: torch.Tensor) -> torch.Tensor:
        if self.percep_type == 'lpips':
            return self.loss_weight * self.lpips_fn(pred_m11, target_m11).mean()

        pred01 = (pred_m11 + 1.0) * 0.5
        target01 = (target_m11 + 1.0) * 0.5
        pred_feat = self.vgg(pred01)
        target_feat = self.vgg(target01.detach())
        loss = pred01.new_zeros(())
        for k, w in self.layer_weights.items():
            loss = loss + float(w) * F.l1_loss(pred_feat[k], target_feat[k])
        return loss * self.loss_weight
