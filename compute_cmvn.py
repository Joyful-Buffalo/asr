#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/27
# @Author  : Joyful Buffalo
# @File    : compute_cmvn.py
"""
基于预计算的 fbank 特征计算全局 CMVN（均值和方差）统计量。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ========== 路径配置 ==========
MANIFEST_PATH = Path(__file__).parent / "dataset/train_fbank_relpath.jsonl"
OUTPUT_PATH = Path(__file__).parent / "dataset/cmvn.npy"
FEAT_ROOT = Path(__file__).parent
# ==============================


def main() -> None:
    # 统计量
    count = 0  # 总帧数
    mean = None
    m2 = None
    num_bins = None

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"正在计算 CMVN，共 {len(lines)} 个样本...")

    for line in tqdm(lines, ncols=100, desc="Computing CMVN"):
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        feat_path = Path(item["feat"])
        if not feat_path.is_absolute():
            feat_path = FEAT_ROOT / feat_path

        # 加载特征
        if feat_path.suffix == ".npy":
            feat = np.load(feat_path)
        else:
            import torch
            feat = torch.load(feat_path, weights_only=True).numpy()

        if num_bins is None:
            num_bins = feat.shape[1]
            mean = np.zeros(num_bins, dtype=np.float64)
            m2 = np.zeros(num_bins, dtype=np.float64)

        # Welford 在线算法
        for frame in feat:
            count += 1
            delta = frame - mean
            mean += delta / count
            delta2 = frame - mean
            m2 += delta * delta2

    variance = m2 / count
    std = np.maximum(np.sqrt(variance), 1e-8)

    # 保存: (2, num_mel_bins)，第一行均值，第二行标准差
    cmvn = np.stack([mean, std], axis=0).astype(np.float32)
    np.save(OUTPUT_PATH, cmvn)

    print(f"\n===== CMVN 统计 =====")
    print(f"总帧数: {count:,}")
    print(f"特征维度: {num_bins}")
    print(f"均值范围: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"标准差范围: [{std.min():.4f}, {std.max():.4f}]")
    print(f"已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
