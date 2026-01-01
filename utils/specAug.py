import torch
from torchaudio.transforms import FrequencyMasking, TimeMasking


class SpecAugment(torch.nn.Module):
    def __init__(
            self,
            freq_mask_param=20,
            num_freq_masks=2,
            time_mask_param=4,
            num_time_masks=2,
            protect_last=False,
    ):
        super().__init__()
        self.protect_last = protect_last
        self.freq_masks = torch.nn.ModuleList([
            FrequencyMasking(freq_mask_param, iid_masks=True) for _ in range(num_freq_masks)
        ])
        self.time_masks = torch.nn.ModuleList([
            TimeMasking(time_mask_param, iid_masks=True) for _ in range(num_time_masks)
        ])

    def forward(self, spec):  # spec: [B, T, 80]
        last = None
        spec = spec.transpose(-2, -1)  # -> [B, 80, T]
        for m in self.freq_masks:  # 频域遮挡
            spec = m(spec)
        if self.protect_last:
            last = spec[:, :, -1:].clone()
            spec = spec[:, :, :-1]
        for m in self.time_masks:
            spec = m(spec)
        if self.protect_last:
            spec = torch.cat([spec, last], dim=-1)
        return spec.transpose(-2, -1)