#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/23
# @Author  : Joyful Buffalo
# @File    : precompute_fbank.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json

import numpy as np
import torch
import torchaudio
from torchaudio.compliance import kaldi
from tqdm import tqdm


@dataclass
class FbankConfig:
    input_manifest: str
    output_manifest: str
    feat_dir: str
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    dither: float = 0.0
    use_numpy: bool = True 


def compute_and_save_fbank(cfg: FbankConfig) -> None:
    """
    读取 input_manifest（每行至少包含 wav + txt/text），
    计算 fbank，存成 .pt，然后写出新的 jsonl：
    {"key": ..., "feat": "...", "feat_frames": 123, "txt": "..."}
    """
    in_path = Path(cfg.input_manifest)
    out_path = Path(cfg.output_manifest)
    feat_root = Path(cfg.feat_dir)
    feat_root.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, ncols=100, desc=f"Fbank {in_path.name}"):
            line = line.strip()
            if not line:
                continue
            item: Dict[str, Any] = json.loads(line)

            wav = item.get("wav")
            txt = item.get("txt") or item.get("text")
            key = item.get("key")
            if wav is None or txt is None:
                raise ValueError("manifest 每行必须至少包含 'wav' 和 'txt' / 'text' 字段")

            wav_path = Path(wav)
            if key is None:
                # 没有提供 key 的话，就用 wav 文件名（不带扩展名）
                key = wav_path.stem

            # 1) 读波形
            waveform, sr = torchaudio.load(wav_path)

            # 2) 多通道转单通道
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 3) 重采样
            if sr != cfg.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=cfg.sample_rate,
                )
                waveform = resampler(waveform)
                sr = cfg.sample_rate

            # 4) Kaldi 风格幅度缩放
            waveform = waveform * (1 << 15)

            # 5) 计算 fbank（参数与训练脚本保持一致）
            feat = kaldi.fbank(
                waveform,
                num_mel_bins=cfg.num_mel_bins,
                frame_length=cfg.frame_length_ms,
                frame_shift=cfg.frame_shift_ms,
                dither=cfg.dither,
                energy_floor=0.0,
                sample_frequency=float(sr),
            )  # (frames, num_mel_bins)

            feat_frames = int(feat.size(0))
            
            # 根据配置选择保存格式
            if cfg.use_numpy:
                feat_path = feat_root / f"{key}.npy"
                # numpy 格式在 WSL2 中加载更快
                np.save(feat_path, feat.numpy())
            else:
                feat_path = feat_root / f"{key}.pt"
                torch.save(feat, feat_path)

            new_item: Dict[str, Any] = {
                "key": key,
                # 用 posix 路径，避免 Windows 反斜杠在别的平台上出事
                "feat": feat_path.as_posix(),
                "feat_frames": feat_frames,
                "txt": txt,
            }
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="离线预提取 fbank，并生成新的 jsonl 清单"
    )
    parser.add_argument(
        "--input-manifest", 
        type=str,
        required=True,
        help="原始 jsonl，包含 wav + txt/text"
        )
    parser.add_argument(
        "--output-manifest", 
        type=str, 
        required=True,
        help="输出 jsonl，包含 feat + feat_frames + txt"
        )
    parser.add_argument(
        "--feat-dir", 
        type=str, 
        required=True,
        help="保存 .pt 特征的目录（会自动创建）"
        )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-mel-bins", type=int, default=80)
    parser.add_argument("--frame-length-ms", type=float, default=25.0)
    parser.add_argument("--frame-shift-ms", type=float, default=10.0)
    parser.add_argument("--dither", type=float, default=0.0)
    parser.add_argument("--use-numpy", action="store_true", default=True,
                        help="使用 numpy 格式保存（默认），在 WSL2 中加载更快")
    parser.add_argument("--use-torch", dest="use_numpy", action="store_false",
                        help="使用 torch 格式保存")

    args = parser.parse_args()

    cfg = FbankConfig(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        feat_dir=args.feat_dir,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
        frame_length_ms=args.frame_length_ms,
        frame_shift_ms=args.frame_shift_ms,
        dither=args.dither,
        use_numpy=args.use_numpy,
    )
    compute_and_save_fbank(cfg)


if __name__ == "__main__":
    main()
