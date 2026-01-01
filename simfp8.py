#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/22
# @Author  : Joyful Buffalo
# @File    : simple_ctc_asr.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterator, cast

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5
from torchaudio.compliance import kaldi
from tqdm import tqdm


@dataclass
class ASRConfig:
    train_manifest: str
    dev_manifest: str
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    dither: float = 0.0
    lstm_hidden_size: int = 256
    lstm_layers: int = 3
    lstm_dropout: float = 0.1
    batch_size: Optional[int] = None
    max_frames_per_batch: Optional[int] = 4096
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 5.0
    num_workers: int = 8
    prefetch_factor: int = 2
    bucket_size: int = 100
    pin_memory: bool = True
    shuffle: bool = True


# ======================
# 动态batch采样器
# ======================
class LengthBucketBatchSampler(Sampler[List[int]]):
    """按长度分桶的动态batch采样器，支持固定batch_size或max_frames_per_batch"""

    def __init__(
            self,
            lengths: List[int],
            batch_size: Optional[int] = None,
            max_frames_per_batch: Optional[int] = None,
            shuffle: bool = True,
            bucket_size: int = 100,
    ):
        super(LengthBucketBatchSampler, self).__init__(None)
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
        # 返回 batch indices
        if self.shuffle:
            indices = [self._indices[i] for i in torch.randperm(len(self._indices), generator=self.g).tolist()]
        else:
            indices = self._indices

        # 分桶 + 桶内按长度升序
        buckets = [
            indices[i:i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]
        for b in buckets:
            b.sort(key=lambda i: self.lengths[i])

        flat = [i for b in buckets for i in b]

        if self.batch_size is not None:
            for i in range(0, len(flat), self.batch_size):
                yield flat[i:i + self.batch_size]
        else:
            current, cur_frames = [], 0
            for i in flat:
                length = int(self.lengths[i])
                if len(current) > 0 and (cur_frames + length) > int(self.max_frames_per_batch):
                    yield current
                    current, cur_frames = [], 0
                current.append(i)
                cur_frames += length
            if len(current) > 0:
                yield current

    def __len__(self) -> int:
        if self.batch_size is not None:
            return math.ceil(len(self.lengths) / self.batch_size)
        # max_frames 模式下，__len__ 很难精确；给出近似上界
        avg = (sum(self.lengths) / max(1, len(self.lengths)))
        return max(1, int(sum(self.lengths) / max(self.max_frames_per_batch, cast(int, avg))))


# ======================
# 字级 tokenizer（0 留给 CTC blank）
# ======================
class CharTokenizer:
    """最简单的“字级”分词器，0 号留给 CTC blank。"""

    def __init__(self, char2id: Dict[str, int]) -> None:
        self.blank_id: int = 0
        self.char2id: Dict[str, int] = char2id
        # id2char 索引 0 为 <blank>，其余按 id 排序
        self.id2char: List[str] = ["<blank>"] + [
            ch for ch, _ in sorted(char2id.items(), key=lambda x: x[1])
        ]

    @classmethod
    def build_from_jsonl(cls, manifest_path: str) -> "CharTokenizer":
        chars: set[str] = set()
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # 支持 txt 或 text 两种字段名
                txt = item.get("txt") or item.get("text")
                if txt is None:
                    raise ValueError("每行 json 需要包含 'txt' 或 'text' 字段")
                for ch in txt:
                    if ch.isspace():
                        continue  # 去掉空格
                    chars.add(ch)
        # 按字典序排序，id 从 1 开始分配，0 保留给 blank
        char2id: Dict[str, int] = {}
        next_id = 1
        for ch in sorted(chars):
            char2id[ch] = next_id
            next_id += 1
        return cls(char2id)

    @property
    def vocab_size(self) -> int:
        # 包含 blank
        return 1 + len(self.char2id)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for ch in text:
            if ch.isspace():
                continue
            idx = self.char2id.get(ch)
            if idx is None:
                # 简单处理：丢弃未登录字（也可以映射到 <unk>，这里先偷懒）
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
    """读取 jsonl 格式清单：{"key":..., "wav":..., "txt":...} 或 {"key":..., "feat":..., "txt":..., "feat_frames":...}"""

    def __init__(self, manifest_path: str, tokenizer: CharTokenizer, config: ASRConfig,
                 compute_lengths: bool = False, use_precomputed_fbank: bool = False) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = config.sample_rate
        self.num_mel_bins = config.num_mel_bins
        self.frame_length_ms = config.frame_length_ms
        self.frame_shift_ms = config.frame_shift_ms
        self.dither = config.dither
        self.compute_lengths = compute_lengths
        self.use_precomputed_fbank = use_precomputed_fbank

        self.entries: List[Dict[str, Any]] = []
        self.lengths: List[int] = []

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                txt = item.get("txt") or item.get("text")
                if txt is None:
                    raise ValueError("清单每行必须包含 'txt'/'text'")

                # 判断是使用预处理的fbank还是原始wav
                if use_precomputed_fbank:
                    feat = item.get("feat")
                    if feat is None:
                        raise ValueError("使用预处理fbank时，清单每行必须包含 'feat'")
                    self.entries.append({"feat": feat, "txt": txt})

                    # 如果需要计算长度（用于动态batch）
                    if compute_lengths:
                        feat_frames = item.get("feat_frames", 0)
                        if feat_frames > 0:
                            self.lengths.append(feat_frames)
                        else:
                            print(f"警告: {feat} 没有帧数信息，使用默认值")
                            self.lengths.append(100)
                else:
                    wav = item.get("wav")
                    if wav is None:
                        raise ValueError("清单每行必须包含 'wav'")
                    self.entries.append({"wav": wav, "txt": txt})

                    # 如果需要计算长度（用于动态batch）
                    if compute_lengths:
                        # 从JSON中读取时长（秒），计算特征帧数
                        duration_sec = item.get("length", 0.0)
                        if duration_sec > 0:
                            duration_ms = duration_sec * 1000
                            feat_frames = int((duration_ms - self.frame_length_ms) / self.frame_shift_ms) + 1
                            self.lengths.append(max(1, feat_frames))
                        else:
                            # 如果没有时长信息，给一个默认值
                            print(f"警告: {wav} 没有时长信息，使用默认值")
                            self.lengths.append(100)

        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 为空")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.entries[idx]
        text: str = item["txt"]

        if self.use_precomputed_fbank:
            # 使用预处理好的fbank特征
            feat_path = item["feat"]
            # 使用 numpy 加载更快（特别是在 WSL2 中）
            if feat_path.endswith('.npy'):
                feat = torch.from_numpy(np.load(feat_path))
            else:
                # 兼容 .pt 格式，使用 weights_only 更快更安全
                feat = torch.load(feat_path, weights_only=False)  # (frames, num_mel_bins)
        else:
            # 实时计算fbank特征
            wav_path = item["wav"]
            waveform, sr = torchaudio.load(wav_path)  # (channels, num_samples)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
                sr = self.sample_rate

            # 与 Kaldi 一致，将 [-1, 1] 量化到 16bit 整数范围
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

        # 文本转为 id 序列
        target_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return feat, target_ids, text


def asr_collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]:
    feats, targets, texts = zip(*batch)

    # 对 fbank 按时间维做 padding
    feat_lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    padded_feats = pad_sequence(feats, batch_first=True)  # (B, T_max, D)

    # 标签拼接成一维向量 + 长度
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    concatenated_targets = torch.cat(targets, dim=0)  # (sum_target,)

    return {
        "feats": padded_feats,
        "feat_lengths": feat_lengths,
        "max_feat_lengths": feat_lengths.max().item(),
        "targets": concatenated_targets,
        "target_lengths": target_lengths,
        "texts": list(texts),
    }


class CTCConformer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            vocab_size: int,
            hidden_size: int = 512,
            num_layers: int = 3,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim * 2)
        self.conformer = torchaudio.models.Conformer(
            input_dim=input_dim * 2,
            num_layers=num_layers,
            ffn_dim=hidden_size,
            dropout=dropout,
            depthwise_conv_kernel_size=15,
            num_heads=8
        )
        self.fc = nn.Linear(input_dim * 2, vocab_size)

    def forward(self, feats: torch.Tensor, feat_lengths, max_length) -> torch.Tensor:
        feats = self.input_proj(feats)
        x, _ = self.conformer(feats, feat_lengths)  # (B, T, 2H)
        logits = self.fc(x)  # (B, T, V)
        return logits


def ctc_greedy_decode(
        logit_batch: torch.Tensor,
        feat_lengths: torch.Tensor,
        blank_id: int,
) -> List[List[int]]:
    """
    最简单 CTC 贪心解码：按时间取 argmax，然后压缩重复 & 去掉 blank。
    logit_batch: (B, T, V)
    feat_lengths: (B,)
    """
    with torch.no_grad():
        probs = F.log_softmax(logit_batch, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)  # (B, T)

    results: List[List[int]] = []
    for b in range(pred_ids.size(0)):
        length = int(feat_lengths[b].item())
        seq = pred_ids[b, :length].tolist()
        collapsed: List[int] = []
        prev = blank_id
        for idx in seq:
            if idx != blank_id and idx != prev:
                collapsed.append(idx)
            prev = idx
        results.append(collapsed)
    return results


def char_edit_distance(ref: str, hyp: str) -> int:
    """标准 Levenshtein 编辑距离（以“字”为单位）"""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    n = len(ref_chars)
    m = len(hyp_chars)
    if n == 0:
        return m
    if m == 0:
        return n
    # DP 矩阵大小 (n+1) x (m+1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # 删除
                dp[i][j - 1] + 1,  # 插入
                dp[i - 1][j - 1] + cost,  # 替换
            )
    return dp[n][m]


def evaluate_cer(
        model: nn.Module,
        dataloader: DataLoader,
        tokenizer: CharTokenizer,
        device: torch.device,
) -> float:
    model.eval()
    total_edit = 0
    total_chars = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=100):
            feats = batch["feats"].to(device)
            feat_lengths = batch["feat_lengths"].to(device)
            texts = batch["texts"]
            max_length = batch["max_feat_lengths"]

            # [FP8_FIX] 推理也在 bf16 autocast 下，避免 float 输入 + bf16 权重的 dtype 冲突

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(feats, feat_lengths, max_length)  # (B, T, V)

            pred_id_seqs = ctc_greedy_decode(logits, feat_lengths, tokenizer.blank_id)

            for ref_text, pred_ids in zip(texts, pred_id_seqs):
                hyp_text = tokenizer.decode_ids(pred_ids)
                # 这里也去掉 ref 中的空白字符
                ref = "".join(ch for ch in ref_text if not ch.isspace())
                edit = char_edit_distance(ref, hyp_text)
                total_edit += edit
                total_chars += len(ref)
    if total_chars == 0:
        return 0.0
    return total_edit / total_chars


# ======================
# 训练 1 个 epoch
# ======================
def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        ctc_loss: nn.CTCLoss,
        device: torch.device,
        max_grad_norm: float,
        tokenizer: CharTokenizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_frames = 0
    total_edit = 0
    total_chars = 0
    idx = 0
    for batch in tqdm(dataloader, ncols=100, position=1):
        feats = batch["feats"].to(device, non_blocking=True)
        feat_lengths = batch["feat_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        max_length = batch["max_feat_lengths"]

        optimizer.zero_grad(set_to_none=True)
        # 训练已经在 bf16 autocast 下，这里保持不变
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(feats, feat_lengths, max_length)  # (B, T, V)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = ctc_loss(log_probs, targets, feat_lengths, target_lengths)
        loss.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_frames = int(feat_lengths.sum().item())
        total_loss += loss.item() * batch_frames
        total_frames += batch_frames
        if idx % 100 == 0:
            with torch.no_grad():
                pred_id_seqs = ctc_greedy_decode(logits, feat_lengths, tokenizer.blank_id)
                offset = 0
                for b, pred_ids in enumerate(pred_id_seqs):
                    cur_len = int(target_lengths[b].item())
                    tgt_ids = targets[offset:offset + cur_len].tolist()
                    offset += cur_len
                    ref_text = tokenizer.decode_ids(tgt_ids)
                    hyp_text = tokenizer.decode_ids(pred_ids)
                    edit = char_edit_distance(ref_text, hyp_text)
                    total_edit += edit
                    total_chars += len(ref_text)
        idx += 1
    if total_chars > 0:
        print(f'train_loss: {total_loss / total_frames:.3f} train cer: {total_edit / total_chars:.3f}')
    if total_frames == 0:
        return 0.0
    return total_loss / total_frames


def dynamic_pre_compile(train_loader, device, model):
    warmup_batch = next(iter(train_loader))
    warmup_feats = warmup_batch["feats"].to(device)  # (B, T, D)
    warmup_feat_lengths = warmup_batch["feat_lengths"].to(device)
    warmup_max_len = int(warmup_feat_lengths.max().item())

    # 标记时间维度为动态
    torch._dynamo.mark_dynamic(warmup_feats, 1, min=1, max=2048)
    model = torch.compile(model)

    # [FP8_FIX] 编译 warmup 也要和训练一样，在 bf16 autocast 里跑
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(warmup_feats, warmup_feat_lengths, warmup_max_len)
        else:
            _ = model(warmup_feats, warmup_feat_lengths, warmup_max_len)
    return model


def main() -> None:
    # bf16+fp8，这里就不强调 TF32 了
    # torch.set_float32_matmul_precision('high')

    # === 配置（这里写死，实际用的时候可以改成 argparse） ===
    config = ASRConfig(
        train_manifest="data/train_fbank_relpath.jsonl",
        dev_manifest="data/dev_fbank_relpath.jsonl",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # === 1. 根据训练集构建"字级"词表 ===
    print("构建字级词表...")
    tokenizer = CharTokenizer.build_from_jsonl(config.train_manifest)
    print("vocab_size (含 blank):", tokenizer.vocab_size)

    # === 2. 构建 Dataset / DataLoader ===
    print("构建数据集...")
    train_dataset = JsonlASRDataset(
        config.train_manifest, tokenizer, config,
        compute_lengths=True, use_precomputed_fbank=True
    )
    dev_dataset = JsonlASRDataset(
        config.dev_manifest, tokenizer, config,
        compute_lengths=True, use_precomputed_fbank=True
    )

    # 创建动态batch采样器
    print("创建动态batch采样器...")
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
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_sampler=dev_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=asr_collate_fn,
        prefetch_factor=config.prefetch_factor,
    )

    # === 3. 构建模型 / 损失 / 优化器 ===
    input_dim = config.num_mel_bins
    vocab_size = tokenizer.vocab_size
    model = CTCConformer(
        input_dim=input_dim,
        vocab_size=vocab_size,
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_layers,
        dropout=config.lstm_dropout,
    ).to(device)

    if not TORCH_VERSION_AT_LEAST_2_5:
        raise RuntimeError("torchao.float8 需要 PyTorch 2.5 及以上版本")

    # 模型参数切到 bf16
    model = model.to(dtype=torch.bfloat16)

    # 只对 in/out 都是 16 对齐的 Linear 做 float8
    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        """
        参考 torchtitan 文档里的 auto_filter_small_kn 思路：
        - 先保证 Linear 的 (in, out) 都是 16 对齐；
        - 再按 (K, N) 过滤掉“小层”，避免在小 GEMM 上用 float8 得不偿失。
        这里 K/N 对应 Linear 的 in_features / out_features。
        """
        # 非 Linear 模块：直接允许交给 convert_to_float8_training 决定
        if not isinstance(mod, nn.Linear):
            return True

        in_features = int(mod.in_features)
        out_features = int(mod.out_features)

        # 1) 基本要求：16 对齐（与官方教程示例一致）
        if in_features % 16 != 0 or out_features % 16 != 0:
            return False

        # 2) 小层过滤（类似 auto_filter_small_kn）：
        #    经验阈值：min(K, N) < 128 或 max(K, N) < 1024 的层，认为规模偏小，关闭 float8。
        #    阈值来自官方推荐“看 performance 表按 GEMM 尺寸筛选”的思路，具体值是这里根据
        #    小模型场景取的保守值，而不是 torchtitan 内部的精确实现。
        k = in_features
        n = out_features
        mn_min = min(k, n)
        mn_max = max(k, n)

        if mn_min < 128 or mn_max < 1024:
            return False

        return True

    model = convert_to_float8_training(
        model,
        module_filter_fn=module_filter_fn,
        config=Float8LinearConfig(pad_inner_dim=True)
    )

    ctc_loss = nn.CTCLoss(
        blank=tokenizer.blank_id,
        reduction="mean",
        zero_infinity=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
    )
    best_cer = 1.0
    best_path = Path("saved_models/best_ctc_asr_cer18.pt")

    epoch = 1

    # 先 float8 转换，再做动态 compile
    model = dynamic_pre_compile(train_loader, device, model)

    # with sdpa_kernel(SDPBackend.MATH):
    while True:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ctc_loss,
            device,
            config.max_grad_norm,
            tokenizer
        )
        cer = evaluate_cer(model, dev_loader, tokenizer, device)
        print(
            f"[Epoch {epoch:02d}] train_loss_per_frame={train_loss:.4f}, "
            f"dev_CER={cer * 100:.2f}%"
        )
        old_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step(cer)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f'Reducing learning rate {old_lr} to {new_lr}')
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
        epoch += 1


if __name__ == "__main__":
    main()
