#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Dict

import argparse
from tqdm import tqdm


def load_transcript(transcript_path: Path) -> Dict[str, str]:
    """
    读取 aishell_transcript_v0.8.txt
    每行格式：UTT_ID 文本...
    """
    transcript: Dict[str, str] = {}
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            utt_id = parts[0]
            # 后面全部拼回中文文本（官方脚本也是这么干的）:contentReference[oaicite:2]{index=2}
            text = "".join(parts[1:])  # 这里直接去掉空格，做成连续汉字
            transcript[utt_id] = text
    return transcript

md = 0
def make_manifest_for_part(
    corpus_root: Path,
    transcript: Dict[str, str],
    part: str,
    out_path: Path,
) -> None:
    """
    为 train/dev/test 生成 jsonl 清单。
    - corpus_root: 包含 data_aishell 的根目录，比如 ./data
    - part: "train" / "dev" / "test"
    - out_path: 输出的 jsonl 路径
    """
    wav_root = corpus_root / "data_aishell" / "wav" / part
    if not wav_root.is_dir():
        raise RuntimeError(f"找不到目录: {wav_root}")

    num_ok = 0
    num_miss = 0

    # 先收集所有 wav 文件路径
    wav_files = sorted(wav_root.rglob("*.wav"))
    num_total = len(wav_files)

    with out_path.open("w", encoding="utf-8") as fout:
        # 使用 tqdm 显示进度条
        for wav_path in tqdm(wav_files, desc=f"处理 {part}", unit="条"):
            utt_id = wav_path.stem  # BAC009S0002W0122 这种

            txt = transcript.get(utt_id)
            if txt is None:
                num_miss += 1
                # 理论上不会缺；缺了就跳过这条
                continue

            # 用绝对路径更稳一点，Windows/Linux 都能读
            wav_abs = wav_path.resolve().as_posix()

            # 读取音频时长（秒）
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
            except Exception as e:
                print(f"警告: 无法读取 {wav_path} 的时长: {e}")
                duration = 0.0

            item = {
                "key": utt_id,
                "wav": wav_abs.replace('C:/data/gudsen/asr/data','/mnt/c/data/gudsen/asr/data'),
                "txt": txt,
                "length": round(duration, 2),
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            num_ok += 1

    print(
        f"[{part}] 总 wav: {num_total}, 成功写入: {num_ok}, "
        f"缺少转写(被跳过): {num_miss}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 AISHELL-1 转成 simple_ctc_asr.py 使用的 jsonl 清单"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="dataset",
        help="包含 data_aishell 的目录（解压后 data_aishell 就在这里）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="jsonl 输出目录（默认为 data，生成 train/dev/test.jsonl）",
    )
    args = parser.parse_args()

    corpus_root = Path(args.corpus_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    transcript_path = corpus_root / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
    if not transcript_path.is_file():
        raise RuntimeError(f"找不到转写文件: {transcript_path}")

    print("读取转写文件:", transcript_path)
    transcript = load_transcript(transcript_path)
    print("转写条数:", len(transcript))

    for part in ["train", "dev", "test"]:
        out_path = out_root / f"{part}.jsonl"
        print(f"生成 {part} 清单 -> {out_path}")
        make_manifest_for_part(corpus_root, transcript, part, out_path)

    print("全部完成！")


if __name__ == "__main__":
    main()
