# utils/vgg_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """
    Memory-friendly VGG Perceptual Loss for 3DGS rendered images.
    
    Features:
        - Default input downsample to 256x256
        - Default layers: conv1_2, conv2_2, conv3_3
        - Supports AMP
        - Input range [0,1], internally normalized to ImageNet
    """
    def __init__(self,
                 model_type='vgg16',
                 layers=('3','8','15'),  # conv1_2, conv2_2, conv3_3
                 downsample_size=(256,256),
                 use_input_norm=True,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.use_input_norm = use_input_norm
        self.layers = layers
        self.downsample_size = downsample_size

        # Load pretrained VGG
        if model_type == 'vgg16':
            vgg_model = models.vgg16(pretrained=True).features
        elif model_type == 'vgg19':
            vgg_model = models.vgg19(pretrained=True).features
        else:
            raise ValueError("model_type must be 'vgg16' or 'vgg19'")

        self.vgg = vgg_model.to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

        # ImageNet mean/std
        if self.use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.criterion = nn.L1Loss()

    def forward(self, input, target, return_per_layer=False):
        """
        Compute memory-friendly perceptual loss.
        
        Args:
            input (torch.Tensor): [B,3,H,W], 0~1
            target (torch.Tensor): [B,3,H,W], 0~1
            return_per_layer (bool): if True, return list of per-layer losses
        """
        # 下采样到固定尺寸
        if self.downsample_size is not None:
            input = F.interpolate(input, size=self.downsample_size, mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=self.downsample_size, mode='bilinear', align_corners=False)

        x, y = input, target

        # ImageNet normalization
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        feat_x, feat_y = [], []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            y = layer(y)
            if name in self.layers:
                feat_x.append(x)
                feat_y.append(y)

        losses = [self.criterion(fx, fy) for fx, fy in zip(feat_x, feat_y)]

        if return_per_layer:
            return losses
        else:
            return sum(losses)
