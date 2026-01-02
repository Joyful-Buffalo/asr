#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import atexit
import json
import math
import multiprocessing as mp  # ===== [MOD-SCHED] 用于跨 DataLoader worker 共享 speed_prob =====
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from torchaudio.compliance import kaldi
from tqdm import tqdm

from models.conformer import Conformer
from utils.specAug import SpecAugment


@dataclass
class ASRConfig:
    # ===== [MOD] 这里改为“原始 wav 清单”，每行至少包含 wav + txt/text；可选 length(秒) 用于估算桶长度 =====
    train_manifest: str = "dataset/train.jsonl"
    dev_manifest: str = "dataset/dev.jsonl"

    cmvn_path: str = "dataset/cmvn.npy"
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    dither: float = 0.0

    # ===== 贴近 WeNet/常见 Conformer Base 形状 =====
    encoder_dim: int = 256
    ffn_dim: int = 2048
    num_heads: int = 4
    layers: int = 12
    dropout: float = 0.1
    use_rope: bool = False

    batch_size: Optional[int] = None
    max_frames_per_batch: Optional[int] = 8192
    num_epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 5.0

    num_workers: int = 8
    prefetch_factor: int = 2
    bucket_size: int = 100
    pin_memory: bool = True
    shuffle: bool = True
    persistent_workers: bool = True

    eval_max_batches: Optional[int] = None
    eval_cer_max_batches: int = 50
    train_cer_log_interval: int = 200

    use_specaug_prob: float = 0.1
    warmup_steps: int = 12000

    # ===== torchaudio 速度扰动增强（SpeedPerturbation）=====
    use_speed_perturb_prob: float = 0.5
    speed_factors: Tuple[float, ...] = (0.9, 1.0, 1.1)
    speed_apply_to_dev: bool = False

    # ===== [MOD-SCHED] 速度扰动随 epoch 递增：从 speed_prob_start 线性爬到 use_speed_perturb_prob =====
    speed_prob_start: float = 0.0
    speed_prob_ramp_epochs: int = 10


# ===== [MOD-SCHED] 跨 DataLoader worker 共享的 float（用于动态调 speed_prob）=====
class SharedFloat:
    def __init__(self, init_value: float) -> None:
        self._v = mp.Value("d", float(init_value), lock=True)

    def set(self, value: float) -> None:
        with self._v.get_lock():
            self._v.value = float(value)

    def get(self) -> float:
        with self._v.get_lock():
            return float(self._v.value)


def _linear_ramp(epoch: int, start: float, end: float, ramp_epochs: int) -> float:
    # epoch 从 1 开始；ramp_epochs<=0 则直接 end
    if ramp_epochs <= 0:
        return float(end)
    t = float(epoch - 1) / float(ramp_epochs)
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return float(start + (end - start) * t)


# ======================
# 动态 batch 采样器
# ======================
class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: List[int],
        batch_size: Optional[int] = None,
        max_frames_per_batch: Optional[int] = None,
        shuffle: bool = True,
        bucket_size: int = 100,
    ):
        super().__init__(None)
        assert (batch_size is None) ^ (max_frames_per_batch is None), \
            "Exactly one of batch_size or max_frames_per_batch should be provided."
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_frames_per_batch = max_frames_per_batch
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        self._indices = list(range(len(lengths)))
        self.g = torch.Generator().manual_seed(torch.seed())

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            indices = [self._indices[i] for i in torch.randperm(len(self._indices), generator=self.g).tolist()]
        else:
            indices = self._indices

        buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        for b in buckets:
            b.sort(key=lambda i: self.lengths[i])

        flat = [i for b in buckets for i in b]

        if self.batch_size is not None:
            for i in range(0, len(flat), self.batch_size):
                yield flat[i:i + self.batch_size]
        else:
            current: List[int] = []
            cur_frames = 0
            max_frames = cast(int, self.max_frames_per_batch)
            for i in flat:
                length = int(self.lengths[i])
                if len(current) > 0 and (cur_frames + length) > max_frames:
                    yield current
                    current, cur_frames = [], 0
                current.append(i)
                cur_frames += length
            if len(current) > 0:
                yield current

    def __len__(self) -> int:
        if self.batch_size is not None:
            return math.ceil(len(self.lengths) / self.batch_size)
        avg = (sum(self.lengths) / max(1, len(self.lengths)))
        return max(1, int(sum(self.lengths) / max(cast(int, self.max_frames_per_batch), cast(int, avg))))


# ======================
# 字级 tokenizer（0 留给 CTC blank）
# ======================
class CharTokenizer:
    def __init__(self, char2id: Dict[str, int]) -> None:
        self.blank_id: int = 0
        self.char2id: Dict[str, int] = char2id
        self.id2char: List[str] = ["<blank>"] + [ch for ch, _ in sorted(char2id.items(), key=lambda x: x[1])]

    @classmethod
    def build_from_jsonl(cls, manifest_path: str) -> "CharTokenizer":
        chars: set[str] = set()
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                txt = item.get("txt") or item.get("text")
                if txt is None:
                    raise ValueError("每行 json 需要包含 'txt' 或 'text' 字段")
                for ch in txt:
                    if ch.isspace():
                        continue
                    chars.add(ch)
        char2id: Dict[str, int] = {}
        next_id = 1
        for ch in sorted(chars):
            char2id[ch] = next_id
            next_id += 1
        return cls(char2id)

    @property
    def vocab_size(self) -> int:
        return 1 + len(self.char2id)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for ch in text:
            if ch.isspace():
                continue
            idx = self.char2id.get(ch)
            if idx is None:
                continue
            ids.append(idx)
        return ids

    def decode_ids(self, ids: List[int]) -> str:
        chars: List[str] = []
        for idx in ids:
            if idx == self.blank_id:
                continue
            if 0 <= idx < len(self.id2char):
                ch = self.id2char[idx]
                if ch not in ("<blank>", "<unk>"):
                    chars.append(ch)
        return "".join(chars)


class JsonlASRDataset(Dataset):
    # ===== [MOD] 不再支持 “feat 预计算文件”；这里直接从 wav 计算 fbank，并可选 speed perturb =====
    def __init__(
        self,
        manifest_path: str,
        tokenizer: CharTokenizer,
        config: ASRConfig,
        compute_lengths: bool = False,
        is_train: bool = False,
        speed_prob_ctrl: Optional[SharedFloat] = None,  # ===== [MOD-SCHED] 共享的 speed_prob 控制器 =====
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = config.sample_rate
        self.num_mel_bins = config.num_mel_bins
        self.frame_length_ms = config.frame_length_ms
        self.frame_shift_ms = config.frame_shift_ms
        self.dither = config.dither
        self.compute_lengths = compute_lengths
        self.is_train = is_train

        # ===== [MOD-SCHED] speed_prob 由共享控制器提供（worker 内实时读取）=====
        self._speed_prob_ctrl = speed_prob_ctrl

        # ===== speed perturb（torchaudio.transforms.SpeedPerturbation）=====
        # 这里不再把 prob 固化在 dataset 构造时；prob 在 __getitem__ 里动态读
        self._speed_enable = bool(is_train or config.speed_apply_to_dev)
        if self._speed_enable and config.use_speed_perturb_prob > 0 and len(config.speed_factors) > 0:
            self.speed_perturb = torchaudio.transforms.SpeedPerturbation(
                orig_freq=self.sample_rate,
                factors=list(config.speed_factors),
            )
        else:
            self.speed_perturb = None

        # ===== 复用 Resample，避免每个样本 new 一个 resampler =====
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        self.entries: List[Dict[str, Any]] = []
        self.lengths: List[int] = []

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                txt = item.get("txt") or item.get("text")
                wav = item.get("wav")
                if txt is None or wav is None:
                    raise ValueError("清单每行必须包含 'wav' + 'txt'/'text'")
                self.entries.append({"wav": wav, "txt": txt})

                if compute_lengths:
                    # 优先用清单里的 length(秒)；没有就给个保底
                    duration_sec = float(item.get("length", 0.0) or 0.0)
                    if duration_sec > 0:
                        duration_ms = duration_sec * 1000.0
                        feat_frames = int((duration_ms - self.frame_length_ms) / self.frame_shift_ms) + 1
                        self.lengths.append(max(1, feat_frames))
                    else:
                        self.lengths.append(100)

        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 为空")

    def __len__(self) -> int:
        return len(self.entries)

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        r = self._resamplers.get(orig_sr)
        if r is None:
            r = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
            self._resamplers[orig_sr] = r
        return r

    def _get_speed_prob(self) -> float:
        # ===== [MOD-SCHED] worker 内实时读取共享 prob；没给 ctrl 就当 0（或可自己改成常量）=====
        if not self._speed_enable:
            return 0.0
        if self._speed_prob_ctrl is None:
            return 0.0
        return float(self._speed_prob_ctrl.get())

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.entries[idx]
        text: str = item["txt"]
        wav_path = item["wav"]

        waveform, sr = torchaudio.load(wav_path)  # (C, T)

        # 多通道转单通道
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 重采样到目标采样率
        if sr != self.sample_rate:
            waveform = self._get_resampler(int(sr))(waveform)
            sr = self.sample_rate

        # ===== 速度扰动增强：在 waveform 上做（再去算 fbank）=====
        speed_prob = self._get_speed_prob()
        if self.speed_perturb is not None and speed_prob > 0.0 and random.random() < speed_prob:
            w1d = waveform.squeeze(0)  # (T,)
            w1d, _ = self.speed_perturb(w1d, None)
            waveform = w1d.unsqueeze(0)

        # Kaldi 风格幅度缩放
        waveform = waveform * (1 << 15)

        feat = kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=self.dither,
            energy_floor=0.0,
            sample_frequency=float(sr),
        )  # (frames, num_mel_bins)

        target_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return feat, target_ids, text


def asr_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]) -> Dict[str, Any]:
    # 过滤空 target（CTC 会炸）
    filtered: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
    for feat, tgt, txt in batch:
        if tgt.numel() == 0:
            continue
        filtered.append((feat, tgt, txt))
    if not filtered:
        return {"_empty": True}

    feats, targets, texts = zip(*filtered)
    feat_lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    padded_feats = pad_sequence(feats, batch_first=True)  # (B, T, D)

    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    concatenated_targets = torch.cat(targets, dim=0)

    return {
        "feats": padded_feats,
        "feat_lengths": feat_lengths,
        "max_feat_lengths": int(feat_lengths.max().item()),
        "targets": concatenated_targets,
        "target_lengths": target_lengths,
        "texts": list(texts),
    }


class Conv2dSubsampling4(nn.Module):
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        out_freq = (((idim - 1) // 2 - 1) // 2)
        self.out = nn.Linear(odim * out_freq, odim)

    @staticmethod
    def _out_len(lens: torch.Tensor) -> torch.Tensor:
        l1 = torch.div(lens - 1, 2, rounding_mode="floor")
        l2 = torch.div(l1 - 1, 2, rounding_mode="floor")
        return torch.clamp(l2, min=0)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)    # (B, C, T', F')
        b, c, t, f = x.size()
        # x = x.transpose(1, 2).contiguous().view(b, t, c * f)  # (B, T', C*F')
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.flatten(2)
        x = self.out(x)  # (B, T', C)

        out_lens = self._out_len(x_lens)
        return x, out_lens


class CTCConformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        encoder_dim: int,
        ffn_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.subsampling = Conv2dSubsampling4(idim=input_dim, odim=encoder_dim)
        self.conformer = torchaudio.models.Conformer(
            input_dim=encoder_dim,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            depthwise_conv_kernel_size=15,
            num_heads=num_heads,
            # use_rope=use_rope,
        )
        self.ctc_linear = nn.Linear(encoder_dim, vocab_size)

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, out_lens = self.subsampling(feats, feat_lengths)
        x, out_lens2 = self.conformer(x, out_lens)
        logits = self.ctc_linear(x)
        return logits, out_lens2


def ctc_greedy_decode_ids(
    logit_batch: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int,
) -> List[List[int]]:
    with torch.no_grad():
        pred = torch.argmax(F.log_softmax(logit_batch, dim=-1), dim=-1)  # (B, T)
    results: List[List[int]] = []
    bsz = pred.size(0)
    for b in range(bsz):
        T = int(lengths[b].item())
        seq = pred[b, :T].tolist()
        collapsed: List[int] = []
        prev = blank_id
        for idx in seq:
            if idx != blank_id and idx != prev:
                collapsed.append(idx)
            prev = idx
        results.append(collapsed)
    return results


def char_edit_distance(ref: str, hyp: str) -> int:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    n = len(ref_chars)
    m = len(hyp_chars)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref_chars[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


@torch.no_grad()
def evaluate_ctc(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: CharTokenizer,
    device: torch.device,
    ctc_loss_fn: nn.CTCLoss,
    cmvn: Optional[torch.Tensor],
    max_batches: Optional[int],
    cer_max_batches: int,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_frames = 0

    total_edit = 0
    total_chars = 0

    blank_cnt = 0
    frame_cnt = 0
    hyp_len_sum = 0
    hyp_cnt = 0

    invalid_cnt = 0
    seen_cnt = 0

    for bi, batch in enumerate(tqdm(dataloader, ncols=100, desc="dev", leave=False)):
        if batch.get("_empty", False):
            continue
        if max_batches is not None and bi >= max_batches:
            break

        feats = batch["feats"].to(device, non_blocking=True)
        feat_lengths = batch["feat_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        texts = batch["texts"]

        if cmvn is not None:
            feats = (feats - cmvn[0]) / cmvn[1]

        logits, out_lens = model(feats, feat_lengths)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)

        invalid = (out_lens < target_lengths).sum().item()
        invalid_cnt += int(invalid)
        seen_cnt += int(out_lens.numel())

        loss = ctc_loss_fn(log_probs, targets, out_lens, target_lengths)
        loss = loss / max(1, feats.size(0))

        batch_frames = int(out_lens.sum().item())
        total_loss += float(loss.item()) * batch_frames
        total_frames += batch_frames

        pred = torch.argmax(logits, dim=-1)
        blank_cnt += int((pred == tokenizer.blank_id).sum().item())
        frame_cnt += int(pred.numel())

        if bi < cer_max_batches:
            pred_id_seqs = ctc_greedy_decode_ids(logits, out_lens, tokenizer.blank_id)
            for ref_text, pred_ids in zip(texts, pred_id_seqs):
                hyp_text = tokenizer.decode_ids(pred_ids)
                ref = "".join(ch for ch in ref_text if not ch.isspace())
                total_edit += char_edit_distance(ref, hyp_text)
                total_chars += len(ref)
                hyp_len_sum += len(hyp_text)
                hyp_cnt += 1

    cer = (total_edit / total_chars) if total_chars > 0 else 0.0
    loss_per_frame = (total_loss / total_frames) if total_frames > 0 else 0.0
    debug = {
        "blank_frac": (blank_cnt / frame_cnt) if frame_cnt > 0 else 0.0,
        "avg_hyp_len": (hyp_len_sum / hyp_cnt) if hyp_cnt > 0 else 0.0,
        "invalid_ratio": (invalid_cnt / seen_cnt) if seen_cnt > 0 else 0.0,
    }
    return cer, loss_per_frame, debug


class WarmupLR:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> float:
        self.step_num += 1
        s = float(self.step_num)
        w = float(self.warmup_steps)
        scale = (w ** 0.5) * min(s ** -0.5, s * (w ** -1.5))
        for lr0, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr0 * scale
        return self.optimizer.param_groups[0]["lr"]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_sched: WarmupLR,
    ctc_loss_fn: nn.CTCLoss,
    device: torch.device,
    max_grad_norm: float,
    tokenizer: CharTokenizer,
    cmvn: Optional[torch.Tensor],
    specaug: Optional[SpecAugment],
    specaug_prob: float,
    cer_log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_frames = 0

    total_edit = 0
    total_chars = 0

    invalid_cnt = 0
    seen_cnt = 0

    # ===== CPU 也能跑：autocast 只在 CUDA 开 =====
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc="train", leave=False)):
        if batch.get("_empty", False):
            continue
        feats = batch["feats"].to(device, non_blocking=True)
        feat_lengths = batch["feat_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        if cmvn is not None:
            feats = (feats - cmvn[0]) / cmvn[1]
        if specaug is not None and random.random() < specaug_prob:
            feats = specaug(feats)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            logits, out_lens = model(feats, feat_lengths)
            log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)

        invalid = (out_lens < target_lengths).sum().item()
        invalid_cnt += int(invalid)
        seen_cnt += int(out_lens.numel())

        loss = ctc_loss_fn(log_probs, targets, out_lens, target_lengths)
        loss = loss / max(1, feats.size(0))

        loss.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_sched.step()

        batch_frames = int(out_lens.sum().item())
        total_loss += float(loss.item()) * batch_frames
        total_frames += batch_frames

        if cer_log_interval > 0 and (step % cer_log_interval == 0):
            with torch.no_grad():
                pred_id_seqs = ctc_greedy_decode_ids(logits, out_lens, tokenizer.blank_id)
                offset = 0
                for b, pred_ids in enumerate(pred_id_seqs):
                    cur_len = int(target_lengths[b].item())
                    tgt_ids = targets[offset:offset + cur_len].tolist()
                    offset += cur_len
                    ref_text = tokenizer.decode_ids(tgt_ids)
                    hyp_text = tokenizer.decode_ids(pred_ids)
                    total_edit += char_edit_distance(ref_text, hyp_text)
                    total_chars += len(ref_text)

    train_loss = (total_loss / total_frames) if total_frames > 0 else 0.0
    train_cer = (total_edit / total_chars) if total_chars > 0 else 0.0
    inv_ratio = (invalid_cnt / seen_cnt) if seen_cnt > 0 else 0.0
    print(f"train_loss: {train_loss:.3f} train_cer(sampled): {train_cer:.3f} invalid_ratio: {inv_ratio:.2%}")
    return train_loss


def dynamic_pre_compile(train_loader: DataLoader, device: torch.device, model: nn.Module) -> nn.Module:
    warmup_batch = next(iter(train_loader))
    if warmup_batch.get("_empty", False):
        return model

    warmup_feats = warmup_batch["feats"].to(device)
    warmup_feat_lengths = warmup_batch["feat_lengths"].to(device)

    # ===== [MOD] 不要 mark_dynamic(min/max)，会触发 brittle 的 modulo guard/ConstraintViolationError
    # 官方建议遇到误导性的 constraint violation 用 maybe_mark_dynamic :contentReference[oaicite:5]{index=5}
    torch._dynamo.maybe_mark_dynamic(warmup_feats, 0)  # B
    torch._dynamo.maybe_mark_dynamic(warmup_feats, 1)  # T

    try:
        # ===== [MOD] dynamic=True：尽量生成动态 kernel，减少反复 recompilation :contentReference[oaicite:6]{index=6}
        model = torch.compile(model, dynamic=True)
        with torch.no_grad():
            _ = model(warmup_feats, warmup_feat_lengths)
        print("[compile] ok (dynamic=True, maybe_mark_dynamic)")
        return model
    except Exception as e:
        # 这里仍然保底不崩：但正常情况下，上面两处改完就不该再 ConstraintViolationError
        print(f"[compile] failed, fallback eager. err={e}")
        return model



def main() -> None:
    mp.freeze_support()  # ===== [MOD-SCHED] Windows 多进程 DataLoader 兼容 =====

    run_dir = Path("logs") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "stdout.log", "w", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, *fs):
            self.fs = list(fs)

        def write(self, s: str):
            for f in self.fs:
                try:
                    if hasattr(f, "closed") and f.closed:
                        continue
                    f.write(s)
                    f.flush()
                except Exception:
                    pass

        def flush(self):
            for f in self.fs:
                try:
                    if hasattr(f, "closed") and f.closed:
                        continue
                    f.flush()
                except Exception:
                    pass

    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception as e:
        src = f"<FAILED TO READ __file__: {e}>"
    print("===== SCRIPT SNAPSHOT BEGIN =====", file=log_f)
    print(src, file=log_f)
    print("===== SCRIPT SNAPSHOT END =====", file=log_f)
    log_f.flush()

    sys.stdout = _Tee(sys.__stdout__, log_f)

    def _close_log():
        try:
            log_f.close()
        except Exception:
            pass

    atexit.register(_close_log)
    print(f"[log] run_dir={run_dir.resolve()}")

    torch.set_float32_matmul_precision("high")

    config = ASRConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    cmvn_path = Path(config.cmvn_path)
    if cmvn_path.exists():
        cmvn = torch.from_numpy(np.load(cmvn_path)).to(device)
        print(f"已加载 CMVN: {cmvn_path}")
    else:
        cmvn = None
        print(f"警告: CMVN 文件不存在 ({cmvn_path})，跳过归一化")

    print("构建字级词表...")
    tokenizer = CharTokenizer.build_from_jsonl(config.train_manifest)
    print("vocab_size (含 blank):", tokenizer.vocab_size)

    # ===== [MOD-SCHED] speed_prob 共享控制器：初始为 speed_prob_start =====
    speed_prob_ctrl = SharedFloat(config.speed_prob_start)

    print("构建数据集（直接从 wav 计算 fbank）...")
    train_dataset = JsonlASRDataset(
        config.train_manifest, tokenizer, config,
        compute_lengths=True,
        is_train=True,
        speed_prob_ctrl=speed_prob_ctrl,  # ===== [MOD-SCHED] 传入共享控制器 =====
    )
    dev_dataset = JsonlASRDataset(
        config.dev_manifest, tokenizer, config,
        compute_lengths=True,
        is_train=config.speed_apply_to_dev,  # 默认 False：dev 不做速度扰动
        speed_prob_ctrl=(speed_prob_ctrl if config.speed_apply_to_dev else None),  # ===== [MOD-SCHED]
    )

    print("创建动态 batch 采样器...")
    train_sampler = LengthBucketBatchSampler(
        lengths=train_dataset.lengths,
        batch_size=config.batch_size,
        max_frames_per_batch=config.max_frames_per_batch,
        shuffle=config.shuffle,
        bucket_size=config.bucket_size,
    )
    dev_sampler = LengthBucketBatchSampler(
        lengths=dev_dataset.lengths,
        batch_size=config.batch_size,
        max_frames_per_batch=config.max_frames_per_batch,
        shuffle=False,
        bucket_size=config.bucket_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=asr_collate_fn,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=(config.persistent_workers and config.num_workers > 0),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_sampler=dev_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=asr_collate_fn,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=(config.persistent_workers and config.num_workers > 0),
    )

    specaug = SpecAugment(
        freq_mask_param=10,
        num_freq_masks=2,
        time_mask_param=50,
        num_time_masks=2,
        protect_last=False,
    )

    model = CTCConformer(
        input_dim=config.num_mel_bins,
        vocab_size=tokenizer.vocab_size,
        encoder_dim=config.encoder_dim,
        ffn_dim=config.ffn_dim,
        num_layers=config.layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        use_rope=config.use_rope,
    ).to(device)

    ctc_loss_fn = nn.CTCLoss(
        blank=tokenizer.blank_id,
        reduction="sum",
        zero_infinity=True,
    )

    # ===== fused=True 只在 CUDA 下用，避免 CPU 环境报错 =====
    fused_ok = (device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=fused_ok,
    )
    lr_sched = WarmupLR(optimizer, warmup_steps=config.warmup_steps)

    best_cer = 1.0
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    best_path = Path(f"saved_models/best_ctc_asr_cer_{time.strftime('%Y%m%d-%H%M%S')}.pt")

    max_T = max(train_dataset.lengths) if train_dataset.lengths else 2048
    min_T = min(train_dataset.lengths) if train_dataset.lengths else 11
    model = dynamic_pre_compile(train_loader, device, model
                                # , max_T=max_T, min_T=min_T
                                )
    with sdpa_kernel(SDPBackend.MATH):
        for epoch in range(1, config.num_epochs + 1):
            # ===== [MOD-SCHED] 每个 epoch 更新 speed_prob（线性爬到 use_speed_perturb_prob）=====
            cur_speed_prob = _linear_ramp(
                epoch=epoch,
                start=float(config.speed_prob_start),
                end=float(config.use_speed_perturb_prob),
                ramp_epochs=int(config.speed_prob_ramp_epochs),
            )
            speed_prob_ctrl.set(cur_speed_prob)

            cur_specaug_prob = min(config.use_specaug_prob + epoch * 0.04, 0.5)
            print(f"[epoch {epoch:02d}] specaug_prob={cur_specaug_prob:.3f} speed_prob={cur_speed_prob:.3f}")

            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                lr_sched=lr_sched,
                ctc_loss_fn=ctc_loss_fn,
                device=device,
                max_grad_norm=config.max_grad_norm,
                tokenizer=tokenizer,
                cmvn=cmvn,
                specaug=specaug,
                specaug_prob=cur_specaug_prob,
                cer_log_interval=config.train_cer_log_interval,  # ===== [MOD-SCHED] 修复变量名 =====
            )

            cer, dev_loss, dbg = evaluate_ctc(
                model=model,
                dataloader=dev_loader,
                tokenizer=tokenizer,
                device=device,
                ctc_loss_fn=ctc_loss_fn,
                cmvn=cmvn,
                max_batches=config.eval_max_batches,
                cer_max_batches=config.eval_cer_max_batches,
            )
            print(
                f"[Epoch {epoch:02d}] dev_loss_per_frame={dev_loss:.4f}, "
                f"dev_CER(sampled)={cer * 100:.2f}% "
                f"[blank_frac={dbg['blank_frac']*100:.2f}%, avg_hyp_len={dbg['avg_hyp_len']:.2f}, invalid_ratio={dbg['invalid_ratio']*100:.2f}%]"
            )

            if cer < best_cer:
                best_cer = cer
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "char2id": tokenizer.char2id,
                        "blank_id": tokenizer.blank_id,
                    },
                    best_path,
                )
                print(f"  CER 改善，已保存到 {best_path} (best_CER={best_cer * 100:.2f}%)")

            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / 1024**2
                resv = torch.cuda.memory_reserved() / 1024**2
                print(f"allocated={alloc:.1f}MB reserved={resv:.1f}MB lr={optimizer.param_groups[0]['lr']:.3e}")


if __name__ == "__main__":
    main()
