"""Independent wavelet classification models, registered alongside TinyViM."""
import copy

import torch
from torch import nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model

from model.tinyvim import Embedding, TinyViM_depth, TinyViM_width, stem
from model.wavelet_block import WaveletBlock

__all__ = ["WaveletTinyViM", "WaveletStage", "WaveletTinyViM_S",
           "WaveletTinyViM_B", "WaveletTinyViM_L"]


def _validate_positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def WaveletStage(dim, depth):
    """Each original block position owns a separate WaveletBlock."""
    _validate_positive_integer(dim, "dim")
    _validate_positive_integer(depth, "depth")
    return nn.Sequential(*(WaveletBlock(dim) for _ in range(depth)))


class WaveletTinyViM(nn.Module):
    def __init__(self, layers, embed_dims, num_classes=1000, distillation=True,
                 down_patch_size=3, down_stride=2, down_pad=1):
        super().__init__()
        if len(layers) != 4 or len(embed_dims) != 4:
            raise ValueError("layers and embed_dims must describe exactly four stages")
        for name, values in (("layers", layers), ("embed_dims", embed_dims)):
            for value in values:
                _validate_positive_integer(value, name)
        if type(num_classes) is not int:
            raise TypeError("num_classes must be an integer")
        if type(distillation) is not bool:
            raise TypeError("distillation must be a bool")
        _validate_positive_integer(down_patch_size, "down_patch_size")
        _validate_positive_integer(down_stride, "down_stride")
        if type(down_pad) is not int or down_pad < 0:
            raise ValueError("down_pad must be a nonnegative integer")
        self.num_classes = num_classes
        self.dist = distillation
        self.patch_embed = stem(3, embed_dims[0])
        network = []
        for index, (dim, depth) in enumerate(zip(embed_dims, layers)):
            network.append(WaveletStage(dim, depth))
            if index < 3:
                network.append(Embedding(patch_size=down_patch_size, stride=down_stride,
                                         padding=down_pad, in_chans=dim,
                                         embed_dim=embed_dims[index + 1]))
        self.network = nn.ModuleList(network)
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        if self.dist:
            self.dist_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self._init_weights(self)

    def _init_weights(self, module):
        # This traversal owns framework initialization and stops at the block boundary.
        # Never pass it to Module.apply, which visits the children before the parent.
        if isinstance(module, WaveletBlock):
            return
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        for child in module.children():
            self._init_weights(child)

    def forward_tokens(self, x):
        for layer in self.network:
            x = layer(x)
        return x

    def to_deploy(self):
        """Return a fused inference copy, preserving this model and optimizer references."""
        deployed = copy.deepcopy(self).eval()
        for module in deployed.modules():
            if isinstance(module, WaveletBlock):
                module.switch_to_deploy()
        return deployed

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("input must be a real floating-point Tensor")
        if x.ndim != 4 or x.shape[1] != 3 or any(size <= 0 for size in x.shape):
            raise ValueError("classification input must be nonempty B×3×H×W")
        if not x.is_cuda:
            raise RuntimeError("WaveletTinyViM requires CUDA for the official Mamba2 execution path")
        x = self.norm(self.forward_tokens(self.patch_embed(x))).flatten(2).mean(-1)
        logits = self.head(x)
        if not self.dist:
            return logits
        dist_logits = self.dist_head(x)
        if self.training:
            return logits, dist_logits
        return (logits + dist_logits) / 2


def _create_wavelet_model(variant, pretrained=False, **kwargs):
    if variant not in TinyViM_width:
        raise ValueError(f"unknown WaveletTinyViM variant: {variant!r}")
    if pretrained:
        raise ValueError("WaveletTinyViM has no pretrained weights; use pretrained=False")
    if "layers" in kwargs or "embed_dims" in kwargs:
        raise ValueError("registered WaveletTinyViM variants have fixed layers and embed_dims")
    model = WaveletTinyViM(layers=list(TinyViM_depth[variant]),
                          embed_dims=list(TinyViM_width[variant]), **kwargs)
    model.default_cfg = {
        "url": "", "num_classes": 1000, "input_size": (3, 224, 224),
        "pool_size": None, "crop_pct": 0.9, "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
    }
    return model


@register_model
def WaveletTinyViM_S(pretrained=False, **kwargs):
    return _create_wavelet_model("S", pretrained=pretrained, **kwargs)


@register_model
def WaveletTinyViM_B(pretrained=False, **kwargs):
    return _create_wavelet_model("B", pretrained=pretrained, **kwargs)


@register_model
def WaveletTinyViM_L(pretrained=False, **kwargs):
    return _create_wavelet_model("L", pretrained=pretrained, **kwargs)
