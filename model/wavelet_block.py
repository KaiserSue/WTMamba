"""Haar, Mamba2 and frequency-domain processing for WaveletTinyViM."""
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["WaveletBlock"]


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_feature(x, channels=None):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("input must be a real floating-point Tensor")
    if x.ndim != 4 or any(size <= 0 for size in x.shape):
        raise ValueError("input must have nonempty B×C×H×W dimensions")
    if channels is not None and x.shape[1] != channels:
        raise ValueError(f"expected {channels} channels, got {x.shape[1]}")


class _HaarPair:
    """Fixed orthonormal Haar transform, with explicit per-call crop metadata."""

    def decompose(self, x):
        _validate_feature(x)
        height, width = x.shape[-2:]
        # Pad odd dimensions at the bottom/right by replicating edge values.
        # Boundary subbands differ from zero padding; match padding before comparing libraries.
        # Keep the original size for cropping after reconstruction.
        x = F.pad(x, (0, width % 2, 0, height % 2), mode="replicate")
        # Each 2x2 patch is arranged as [[a, b], [c, d]].
        a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
        c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        # Orthonormal Haar uses 1D coefficients +/-1/sqrt(2); their 2D outer
        # products give +/-1/2, hence division by 2 below.
        # Fixed order: LL, differences along width, differences along height,
        # and diagonal detail. This module calls the last three LH, HL, HH;
        # match actual filter directions rather than labels across libraries.
        # PyWavelets dwt2 order (cA, cH, cV, cD) maps to indices [0, 2, 1, 3].
        # This order also selects the high-frequency branches; changing it
        # changes the behavior of existing trained weights.
        return ((a + b + c + d) / 2,
                (a - b + c - d) / 2,
                (a + b - c - d) / 2,
                (a - b - c + d) / 2), (height, width)

    def reconstruct(self, subbands, original_hw):
        if len(subbands) != 4:
            raise ValueError("Haar reconstruction requires LL, LH, HL, HH")
        # Use decompose's order; map external subbands by filter direction first.
        ll, lh, hl, hh = subbands
        _validate_feature(ll)
        if any(band.shape != ll.shape or band.dtype != ll.dtype or
               band.device != ll.device for band in subbands):
            raise ValueError("all Haar subbands must have identical shape, dtype and device")
        if len(original_hw) != 2 or any(type(size) is not int or size <= 0
                                        for size in original_hw):
            raise ValueError("original_hw must contain two positive integers")
        height, width = original_hw
        if 2 * ll.shape[-2] - height not in (0, 1) or 2 * ll.shape[-1] - width not in (0, 1):
            raise ValueError("original_hw must match the Haar subbands before padding")
        # The paired inverse restores [[a, b], [c, d]], then interleaves pixels.
        a, b = (ll + lh + hl + hh) / 2, (ll - lh + hl - hh) / 2
        c, d = (ll + lh - hl - hh) / 2, (ll - lh - hl + hh) / 2
        top = torch.stack((a, b), dim=-1).flatten(-2)
        bottom = torch.stack((c, d), dim=-1).flatten(-2)
        result = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
        # Remove the bottom/right padding added for odd input dimensions.
        return result[..., :height, :width]


class _ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        _positive_integer(channels, "channels")
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            return x * torch.sigmoid(self.weight * x + self.bias)


class _RepConv(nn.Module):
    """Depthwise 3×3 / 1×1 / identity branches, fused to one depthwise 3×3."""

    def __init__(self, channels):
        super().__init__()
        _positive_integer(channels, "channels")
        self.deploy = False
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels))
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels))
        self.identity_bn = nn.BatchNorm2d(channels)
        for branch in (self.branch3, self.branch1):
            nn.init.trunc_normal_(branch[0].weight, std=0.02)

    def forward(self, x):
        if self.deploy:
            return self.reparam(x)
        return self.branch3(x) + self.branch1(x) + self.identity_bn(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return self
        self.reparam = self.fuse()
        del self.branch3, self.branch1, self.identity_bn
        self.deploy = True
        self.eval()
        return self

    @staticmethod
    def _fold_bn(weight, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return weight * scale[:, None, None, None], bn.bias - scale * bn.running_mean

    @torch.no_grad()
    def fuse(self):
        if self.deploy:
            return self.reparam
        w3, b3 = self._fold_bn(self.branch3[0].weight, self.branch3[1])
        w1, b1 = self._fold_bn(self.branch1[0].weight, self.branch1[1])
        channels = w3.shape[0]
        identity = torch.ones(channels, 1, 1, 1, device=w3.device, dtype=w3.dtype)
        wi, bi = self._fold_bn(identity, self.identity_bn)
        fused = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True,
                          device=w3.device, dtype=w3.dtype)
        fused.weight.copy_(w3 + F.pad(w1 + wi, (1, 1, 1, 1)))
        fused.bias.copy_(b3 + b1 + bi)
        return fused.eval()


def _hermitian_projection(spectrum):
    height, width = spectrum.shape[-2:]
    rows = (-torch.arange(height, device=spectrum.device)) % height
    columns = (-torch.arange(width, device=spectrum.device)) % width
    partner = spectrum.index_select(-2, rows).index_select(-1, columns).conj()
    return (spectrum + partner) / 2


class _FrequencyBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.repconv = _RepConv(2 * dim)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
            packed = torch.cat((spectrum.real, spectrum.imag), dim=1)
            real, imag = self.repconv(packed).chunk(2, dim=1)
            spectrum = _hermitian_projection(torch.complex(real, imag))
            return torch.fft.ifft2(spectrum, dim=(-2, -1), norm="ortho").real


class _LowFrequencyMamba2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Import errors propagate unchanged, preserving the exact missing dependency.
        # Delay importing Mamba2 so registration does not load its optional stack.
        import_module("causal_conv1d")  # Required by the selected fused Mamba2 path.
        from mamba_ssm import Mamba2

        self.dim = dim
        self.inner_dim = 32 * ((dim + 31) // 32)
        self.in_adapter = nn.Identity()
        self.out_adapter = nn.Identity()
        if dim != self.inner_dim:
            self.in_adapter = nn.Linear(dim, self.inner_dim, bias=False)
            self.out_adapter = nn.Linear(self.inner_dim, dim, bias=False)
            nn.init.trunc_normal_(self.in_adapter.weight, std=0.02)
            nn.init.trunc_normal_(self.out_adapter.weight, std=0.02)
        # D is a multiple of 32, so headdim=8 makes nheads=2D/8 a multiple of 8.
        # This aligns the fused causal-conv1d projection stride without widening D.
        self.mamba = Mamba2(d_model=self.inner_dim, d_state=128, d_conv=4,
                            expand=2, headdim=8, ngroups=1, chunk_size=256,
                            use_mem_eff_path=True)

    def forward(self, x):
        batch, channels, height, width = x.shape
        # Keep the adapters and fused SSM in FP32 even under outer CUDA AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            sequence = x.float().flatten(2).transpose(1, 2).contiguous()
            sequence = self.out_adapter(self.mamba(self.in_adapter(sequence)))
            return sequence.transpose(1, 2).reshape(batch, channels, height, width)


class WaveletBlock(nn.Module):
    """WT → low Mamba2 / three frequency branches → gates → IWT.

    Parameters, wavelet arithmetic and the low mixer stay FP32 under outer AMP.
    The production Mamba2 path requires CUDA. No residual or FFN is added.
    """

    def __init__(self, dim):
        super().__init__()
        _positive_integer(dim, "dim")
        self.dim = dim
        self.haar = _HaarPair()
        self.low_mixer = _LowFrequencyMamba2(dim)
        self.high_branches = nn.ModuleList([_FrequencyBranch(dim) for _ in range(3)])
        self.low_gate = _ChannelGate(dim)
        self.high_gate = _ChannelGate(3 * dim)

    def switch_to_deploy(self):
        """Fuse owned frequency branches for inference; this transition is irreversible."""
        for branch in self.high_branches:
            branch.repconv.switch_to_deploy()
        self.eval()
        return self

    def forward(self, x):
        _validate_feature(x, self.dim)
        if not x.is_cuda:
            raise RuntimeError("WaveletBlock requires CUDA for the official Mamba2 execution path")
        input_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            subbands, original_hw = self.haar.decompose(x.float())
        # The low mixer owns its FP32 policy for every caller.
        low = self.low_mixer(subbands[0])
        with torch.autocast(device_type="cuda", enabled=False):
            low = self.low_gate(low.float())
            high = torch.cat([branch(band) for branch, band in
                              zip(self.high_branches, subbands[1:])], dim=1)
            high_bands = self.high_gate(high).chunk(3, dim=1)
            result = self.haar.reconstruct((low, *high_bands), original_hw)
            return result.to(dtype=input_dtype)
