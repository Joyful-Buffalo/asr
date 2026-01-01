#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计 train/dev/test 数据集的音频长度分布
按 0.5 秒为一个桶进行统计
"""
from __future__ import annotations

import wave
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm


def analyze_duration_distribution(corpus_root: Path, part: str, bin_size: float = 0.5):
    """
    分析指定数据集的音频长度分布
    
    Args:
        corpus_root: 包含 data_aishell 的根目录
        part: "train" / "dev" / "test"
        bin_size: 桶的大小（秒），默认0.5秒
    
    Returns:
        dict: 长度分布统计结果
    """
    wav_root = corpus_root / "data_aishell" / "wav" / part
    if not wav_root.is_dir():
        raise RuntimeError(f"找不到目录: {wav_root}")
    
    # 收集所有 wav 文件
    wav_files = sorted(wav_root.rglob("*.wav"))
    
    # 统计信息
    duration_bins = defaultdict(int)  # 桶 -> 数量
    durations = []  # 记录所有时长
    error_count = 0
    
    print(f"\n开始分析 {part} 数据集...")
    for wav_path in tqdm(wav_files, desc=f"读取 {part}", unit="个文件"):
        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                durations.append(duration)
                
                # 计算属于哪个桶 (向下取整)
                bin_idx = int(duration / bin_size)
                duration_bins[bin_idx] += 1
        except Exception as e:
            error_count += 1
            # print(f"警告: 无法读取 {wav_path}: {e}")
    
    return {
        'durations': durations,
        'bins': duration_bins,
        'total_files': len(wav_files),
        'error_count': error_count,
        'bin_size': bin_size
    }


def print_statistics(part: str, stats: dict):
    """打印统计结果"""
    durations = stats['durations']
    bins = stats['bins']
    bin_size = stats['bin_size']
    
    if not durations:
        print(f"\n{part} 数据集无有效数据")
        return
    
    # 基本统计
    total_files = len(durations)
    total_duration = sum(durations)
    avg_duration = total_duration / total_files
    min_duration = min(durations)
    max_duration = max(durations)
    
    print(f"\n{'='*60}")
    print(f"{part.upper()} 数据集统计")
    print(f"{'='*60}")
    print(f"总文件数: {total_files}")
    print(f"总时长: {total_duration:.2f} 秒 ({total_duration/3600:.2f} 小时)")
    print(f"平均时长: {avg_duration:.2f} 秒")
    print(f"最短时长: {min_duration:.2f} 秒")
    print(f"最长时长: {max_duration:.2f} 秒")
    
    if stats['error_count'] > 0:
        print(f"读取错误: {stats['error_count']} 个文件")
    
    # 打印分布（按桶排序）
    print(f"\n长度分布 (桶大小={bin_size}秒):")
    print(f"{'时长区间':<20} {'数量':>10} {'占比':>10} {'累积占比':>10} {'可视化'}")
    print("-" * 70)
    
    sorted_bins = sorted(bins.items())
    cumulative = 0
    
    for bin_idx, count in sorted_bins:
        start = bin_idx * bin_size
        end = (bin_idx + 1) * bin_size
        percentage = (count / total_files) * 100
        cumulative += count
        cumulative_pct = (cumulative / total_files) * 100
        
        # 简单的可视化条形图（每个#代表1%）
        bar = '█' * int(percentage)
        
        print(f"[{start:4.1f}s - {end:4.1f}s)  {count:>10}  {percentage:>8.2f}%  {cumulative_pct:>8.2f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="统计 AISHELL-1 数据集的音频长度分布"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data",
        help="包含 data_aishell 的目录",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.5,
        help="桶的大小（秒），默认0.5秒",
    )
    args = parser.parse_args()
    
    corpus_root = Path(args.corpus_dir)
    bin_size = args.bin_size
    
    # 分析三个数据集
    for part in ["train", "dev", "test"]:
        try:
            stats = analyze_duration_distribution(corpus_root, part, bin_size)
            print_statistics(part, stats)
        except Exception as e:
            print(f"\n处理 {part} 时出错: {e}")
    
    print(f"\n{'='*60}")
    print("分析完成！")


if __name__ == "__main__":
    main()


