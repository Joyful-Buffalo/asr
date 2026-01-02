#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
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

import re

try:
    from pyctcdecode import build_ctcdecoder
except Exception:
    build_ctcdecoder = None

@dataclass
class TestConfig:
    ckpt_path: str = "saved_models/best_ctc_asr_cer_20260102-103246.pt"
    test_manifest: str = "dataset/test.jsonl"
    cmvn_path: str = "dataset/cmvn.npy"

    batch_size: Optional[int] = None
    max_frames_per_batch: int = 12000
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle: bool = False
    bucket_size: int = 100

    use_pyctcdecode_beam: bool = True
    beam_width: int = 5

    kenlm_model_path: str = "/home/lhc/data/gudsen/asr/zh_giga.no_cna_cmn.prune01244.klm"
    lm_alpha: float = 0.8
    lm_beta: float = 0.0

    hotwords: Tuple[str, ...] = ()
    hotword_weight: float = 10.0

    output_jsonl: str = "results_test.jsonl"
    max_utts: Optional[int] = None
    print_examples: int = 50

    device: str = "cuda"
    use_compile: bool = False
    force_sdpa_math: bool = True


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
# tokenizer
# ======================
class CharTokenizer:
    def __init__(self, char2id: Dict[str, int], blank_id: int = 0) -> None:
        self.blank_id = int(blank_id)
        self.char2id = dict(char2id)
        inv = sorted(self.char2id.items(), key=lambda x: x[1])
        self.id2char: List[str] = ["<blank>"] + [ch for ch, _ in inv]

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
        out: List[str] = []
        for idx in ids:
            if idx == self.blank_id:
                continue
            if 0 <= idx < len(self.id2char):
                ch = self.id2char[idx]
                if ch not in ("<blank>", "<unk>"):
                    out.append(ch)
        return "".join(out)


# ======================
# Dataset：复现训练时的 fbank
# ======================
class JsonlASRTestDataset(Dataset):
    def __init__(self, manifest_path: str, cfg_from_ckpt: Dict[str, Any], tokenizer: CharTokenizer) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer

        self.sample_rate = int(cfg_from_ckpt.get("sample_rate", 16000))
        self.num_mel_bins = int(cfg_from_ckpt.get("num_mel_bins", 80))
        self.frame_length_ms = float(cfg_from_ckpt.get("frame_length_ms", 25.0))
        self.frame_shift_ms = float(cfg_from_ckpt.get("frame_shift_ms", 10.0))
        self.dither = float(cfg_from_ckpt.get("dither", 0.0))

        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        self.entries: List[Dict[str, Any]] = []
        self.lengths: List[int] = []

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                wav = item.get("wav")
                txt = item.get("txt") or item.get("text")
                if wav is None:
                    raise ValueError("test 清单每行必须包含 'wav'")
                self.entries.append({"wav": wav, "txt": txt})

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.entries[idx]
        wav_path = item["wav"]
        ref_text = item.get("txt")

        waveform, sr = torchaudio.load(wav_path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if int(sr) != self.sample_rate:
            waveform = self._get_resampler(int(sr))(waveform)
            sr = self.sample_rate

        waveform = waveform * (1 << 15)

        feat = kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=self.dither,
            energy_floor=0.0,
            sample_frequency=float(sr),
        )

        if isinstance(ref_text, str) and len(ref_text) > 0:
            tgt = torch.tensor(self.tokenizer.encode(ref_text), dtype=torch.long)
        else:
            tgt = torch.empty((0,), dtype=torch.long)

        return {
            "feat": feat,
            "feat_len": int(feat.size(0)),
            "tgt": tgt,
            "tgt_len": int(tgt.numel()),
            "wav": wav_path,
            "ref": ref_text if isinstance(ref_text, str) else "",
        }


def test_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    feats = [b["feat"] for b in batch]
    feat_lens = torch.tensor([b["feat_len"] for b in batch], dtype=torch.long)
    padded_feats = pad_sequence(feats, batch_first=True)

    tgts = [b["tgt"] for b in batch]
    tgt_lens = torch.tensor([b["tgt_len"] for b in batch], dtype=torch.long)
    concat_tgts = torch.cat(tgts, dim=0) if any(t.numel() > 0 for t in tgts) else torch.empty((0,), dtype=torch.long)

    return {
        "feats": padded_feats,
        "feat_lengths": feat_lens,
        "targets": concat_tgts,
        "target_lengths": tgt_lens,
        "wavs": [b["wav"] for b in batch],
        "refs": [b["ref"] for b in batch],
    }


# ======================
# Model：复现训练结构
# ======================
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
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3).contiguous().flatten(2)
        x = self.out(x)
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
        )
        self.ctc_linear = nn.Linear(encoder_dim, vocab_size)

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, out_lens = self.subsampling(feats, feat_lengths)
        x, out_lens2 = self.conformer(x, out_lens)
        logits = self.ctc_linear(x)
        return logits, out_lens2


# ======================
# Decoding + metric
# ======================
def ctc_greedy_decode_ids(logits: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> List[List[int]]:
    pred = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)
    out: List[List[int]] = []
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
        out.append(collapsed)
    return out


def _norm_for_cer(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def _split_for_wer(s: str) -> List[str]:
    s2 = s.strip()
    if " " in s2:
        return [w for w in s2.split() if w]
    s2 = _norm_for_cer(s2)
    return list(s2)


def _edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
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
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


# ===== [MOD] 修复点在这里：wrapper 内部把 blank 的 label 变成 ' '，阻止 pyctcdecode 自动补空格导致 vocab+1
# 同时保留 pad 兜底，彻底杜绝 4231 vs 4232
class PyCTCDecodeWrapper:
    def __init__(
        self,
        labels: List[str],
        blank_id: int,                 # ===== [MOD] 新增参数 =====
        kenlm_model_path: str,
        alpha: float,
        beta: float,
    ) -> None:
        self._decoder = None
        self._ok = False
        self._why = ""
        self._labels: List[str] = list(labels)

        try:
            if build_ctcdecoder is None:
                raise RuntimeError("pyctcdecode 未安装：pip install pyctcdecode")

            if not (0 <= int(blank_id) < len(self._labels)):
                raise ValueError(f"blank_id 越界: {blank_id}, len(labels)={len(self._labels)}")

            # ===== [MOD] 关键：把 blank 显示成空格（只改显示，不改 logits 维度）
            self._labels[int(blank_id)] = " "

            km = kenlm_model_path.strip()
            if km:
                self._decoder = build_ctcdecoder(self._labels, kenlm_model_path=km, alpha=alpha, beta=beta)
            else:
                self._decoder = build_ctcdecoder(self._labels)

            self._ok = True
        except Exception as e:
            self._decoder = None
            self._ok = False
            self._why = str(e)

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def why_not(self) -> str:
        return self._why

    def decode_one(
        self,
        logits_2d: np.ndarray,
        hotwords: Tuple[str, ...],
        hotword_weight: float,
        beam_width: int,
    ) -> str:
        if not self._ok or self._decoder is None:
            return ""

        l2d = np.asarray(logits_2d, dtype=np.float32)
        if l2d.ndim != 2:
            raise ValueError(f"logits_2d 必须是 2D，实际 shape={l2d.shape}")

        try:
            if hotwords:
                return cast(
                    str,
                    self._decoder.decode(
                        l2d,
                        hotwords=list(hotwords),
                        hotword_weight=float(hotword_weight),
                        beam_width=int(beam_width),
                    ),
                )
            return cast(str, self._decoder.decode(l2d, beam_width=int(beam_width)))
        except ValueError as e:
            # ===== [MOD] 兜底：如果内部仍然认为 vocab=V+1，就 pad 一列极小值再 decode
            msg = str(e)
            m_vocab = re.search(r"vocabulary is size (\d+)", msg)
            if m_vocab is None:
                raise
            vocab_need = int(m_vocab.group(1))
            if vocab_need == l2d.shape[1] + 1:
                pad_col = np.full((l2d.shape[0], 1), -1.0e9, dtype=l2d.dtype)
                l2d_pad = np.concatenate([l2d, pad_col], axis=1)
                if hotwords:
                    return cast(
                        str,
                        self._decoder.decode(
                            l2d_pad,
                            hotwords=list(hotwords),
                            hotword_weight=float(hotword_weight),
                            beam_width=int(beam_width),
                        ),
                    )
                return cast(str, self._decoder.decode(l2d_pad, beam_width=int(beam_width)))
            raise


def _simple_token_confidence(logits: torch.Tensor, out_lens: torch.Tensor, blank_id: int) -> List[float]:
    logp = F.log_softmax(logits.float(), dim=-1)
    pred = torch.argmax(logp, dim=-1)
    confs: List[float] = []
    for b in range(logits.size(0)):
        T = int(out_lens[b].item())
        p = pred[b, :T]
        lp = logp[b, :T]
        mask = (p != blank_id)
        if mask.any():
            confs.append(float(lp[mask, p[mask]].mean().item()))
        else:
            confs.append(float(lp[:, blank_id].mean().item()))
    return confs


def _rough_alignment_from_argmax(
    logits: torch.Tensor,
    out_lens: torch.Tensor,
    blank_id: int,
    frame_shift_ms: float,
    subsample_factor: int = 4,
) -> List[List[Dict[str, Any]]]:
    pred = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)
    out: List[List[Dict[str, Any]]] = []
    for b in range(pred.size(0)):
        T = int(out_lens[b].item())
        seq = pred[b, :T].tolist()
        items: List[Dict[str, Any]] = []
        prev = blank_id
        for t, idx in enumerate(seq):
            if idx != blank_id and idx != prev:
                time_sec = (t * subsample_factor * frame_shift_ms) / 1000.0
                items.append({"token_id": int(idx), "time_sec": float(time_sec)})
            prev = idx
        out.append(items)
    return out


# ===== [MOD-LOAD] 这里就是你缺的那一步：剥离 torch.compile / DDP 常见前缀 =====
def _strip_prefix(state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def normalize_state_dict(state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    if not state:
        return state, "empty"

    keys = list(state.keys())

    for p in ("_orig_mod.", "module."):
        if all(k.startswith(p) for k in keys):
            return _strip_prefix(state, p), f"strip_all:{p}"

    for p in ("_orig_mod.", "module."):
        n = sum(1 for k in keys if k.startswith(p))
        if n >= int(0.9 * len(keys)):
            return _strip_prefix(state, p), f"strip_most({n}/{len(keys)}):{p}"

    return state, "no_strip"


def load_checkpoint(ckpt_path: str) -> Tuple[Dict[str, Any], Dict[str, int], int, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    train_cfg = cast(Dict[str, Any], ckpt.get("config", {}))
    char2id = cast(Dict[str, int], ckpt.get("char2id", {}))
    blank_id = int(ckpt.get("blank_id", 0))

    state = ckpt.get("model_state_dict", None)
    if state is None:
        state = ckpt.get("state_dict", None)
    if state is None:
        state = ckpt.get("model", None)
    model_state = cast(Dict[str, Any], state if isinstance(state, dict) else {})

    if not char2id:
        raise RuntimeError("ckpt 里没找到 char2id；请确认你保存时包含它。")
    if not model_state:
        raise RuntimeError("ckpt 里没找到 model_state_dict/state_dict；请确认你保存时包含它。")

    model_state, info = normalize_state_dict(model_state)
    print(f"[ckpt] state_dict normalize: {info}")

    return train_cfg, char2id, blank_id, model_state


def main() -> None:
    cfg = TestConfig()

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[env] device={device}, torch={torch.__version__}, torchaudio={torchaudio.__version__}")

    torch.set_float32_matmul_precision("high")

    train_cfg, char2id, blank_id, model_state = load_checkpoint(cfg.ckpt_path)
    tokenizer = CharTokenizer(char2id, blank_id=blank_id)
    print(f"[ckpt] vocab_size={tokenizer.vocab_size}, blank_id={tokenizer.blank_id}")

    cmvn = None
    cmvn_path = Path(cfg.cmvn_path)
    if cmvn_path.exists():
        cmvn = torch.from_numpy(np.load(cmvn_path)).to(device)
        print(f"[cmvn] loaded: {cmvn_path}")
    else:
        print(f"[cmvn] not found, skip: {cmvn_path}")

    ds = JsonlASRTestDataset(cfg.test_manifest, train_cfg, tokenizer)
    if cfg.max_utts is not None:
        ds.entries = ds.entries[: int(cfg.max_utts)]
        ds.lengths = ds.lengths[: int(cfg.max_utts)]

    sampler = LengthBucketBatchSampler(
        lengths=ds.lengths,
        batch_size=cfg.batch_size,
        max_frames_per_batch=cfg.max_frames_per_batch,
        shuffle=cfg.shuffle,
        bucket_size=cfg.bucket_size,
    )

    dl = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=test_collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
    )
    print(f"[data] utts={len(ds)}")

    model = CTCConformer(
        input_dim=int(train_cfg.get("num_mel_bins", 80)),
        vocab_size=tokenizer.vocab_size,
        encoder_dim=int(train_cfg.get("encoder_dim", 256)),
        ffn_dim=int(train_cfg.get("ffn_dim", 2048)),
        num_layers=int(train_cfg.get("layers", 12)),
        num_heads=int(train_cfg.get("num_heads", 4)),
        dropout=float(train_cfg.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(model_state, strict=True)
    model.eval()

    if cfg.use_compile:
        try:
            model = torch.compile(model, dynamic=True)
            print("[compile] enabled (inference)")
        except Exception as e:
            print(f"[compile] failed, continue eager: {e}")

    beam_decoder: Optional[PyCTCDecodeWrapper] = None
    if cfg.use_pyctcdecode_beam:
        # ===== [MOD] labels 直接用 tokenizer.id2char；空格修复在 wrapper 内做（blank -> ' '）
        labels = list(tokenizer.id2char)  # len == vocab_size
        beam_decoder = PyCTCDecodeWrapper(
            labels=labels,
            blank_id=tokenizer.blank_id,  # ===== [MOD] 传入 blank_id
            kenlm_model_path=cfg.kenlm_model_path,
            alpha=cfg.lm_alpha,
            beta=cfg.lm_beta,
        )
        if beam_decoder.ok:
            print("[beam] pyctcdecode ready "
                  f"(kenlm={'on' if cfg.kenlm_model_path.strip() else 'off'}, beam_width={cfg.beam_width})")
        else:
            print(f"[beam] pyctcdecode not available -> greedy only. err={beam_decoder.why_not}")
            beam_decoder = None

    out_path = Path(cfg.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fo = out_path.open("w", encoding="utf-8")

    cer_edit_sum_g = 0
    cer_len_sum = 0
    wer_edit_sum_g = 0
    wer_len_sum = 0
    cer_edit_sum_b = 0
    wer_edit_sum_b = 0

    blank_cnt = 0
    frame_cnt = 0
    hyp_len_sum_g = 0
    hyp_len_sum_b = 0
    n_print = 0

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()

    with torch.inference_mode():
        kernel_ctx = sdpa_kernel(SDPBackend.MATH) if cfg.force_sdpa_math else torch.no_grad()
        with kernel_ctx:
            for batch in tqdm(dl, ncols=110, desc="test"):
                feats = batch["feats"].to(device, non_blocking=True)
                feat_lengths = batch["feat_lengths"].to(device, non_blocking=True)
                refs: List[str] = batch["refs"]
                wavs: List[str] = batch["wavs"]

                if cmvn is not None:
                    feats = (feats - cmvn[0]) / cmvn[1]

                with amp_ctx:
                    logits, out_lens = model(feats, feat_lengths)

                pred = torch.argmax(logits, dim=-1)
                blank_cnt += int((pred == tokenizer.blank_id).sum().item())
                frame_cnt += int(pred.numel())

                greedy_ids = ctc_greedy_decode_ids(logits, out_lens, tokenizer.blank_id)
                greedy_texts = [tokenizer.decode_ids(ids) for ids in greedy_ids]
                hyp_len_sum_g += sum(len(t) for t in greedy_texts)

                beam_texts: List[str] = [""] * len(greedy_texts)
                if beam_decoder is not None and beam_decoder.ok:
                    for i in range(len(greedy_texts)):
                        T = int(out_lens[i].item())
                        l2d = logits[i, :T].detach().float().cpu().numpy()
                        beam_texts[i] = beam_decoder.decode_one(
                            logits_2d=l2d,
                            hotwords=cfg.hotwords,
                            hotword_weight=cfg.hotword_weight,
                            beam_width=cfg.beam_width,
                        )
                    hyp_len_sum_b += sum(len(t) for t in beam_texts)

                confs = _simple_token_confidence(logits, out_lens, tokenizer.blank_id)
                aligns = _rough_alignment_from_argmax(
                    logits=logits,
                    out_lens=out_lens,
                    blank_id=tokenizer.blank_id,
                    frame_shift_ms=float(train_cfg.get("frame_shift_ms", 10.0)),
                    subsample_factor=4,
                )

                for i, (wav, ref, hyp_g) in enumerate(zip(wavs, refs, greedy_texts)):
                    hyp_b = beam_texts[i] if (beam_decoder is not None and beam_decoder.ok) else ""

                    ref_norm = _norm_for_cer(ref) if ref else ""
                    hyp_g_norm = _norm_for_cer(hyp_g)

                    sample: Dict[str, Any] = {
                        "wav": wav,
                        "ref": ref,
                        "hyp_greedy": hyp_g,
                        "hyp_beam": hyp_b,
                        "conf_logp": confs[i],
                        "align_greedy": aligns[i],
                    }

                    if ref_norm:
                        cer_edit_g = _edit_distance(list(ref_norm), list(hyp_g_norm))
                        cer_edit_sum_g += cer_edit_g
                        cer_len_sum += len(ref_norm)

                        ref_w = _split_for_wer(ref)
                        hyp_g_w = _split_for_wer(hyp_g)
                        wer_edit_g = _edit_distance(ref_w, hyp_g_w)
                        wer_edit_sum_g += wer_edit_g
                        wer_len_sum += len(ref_w)

                        sample["cer_greedy"] = cer_edit_g / max(1, len(ref_norm))
                        sample["wer_greedy"] = wer_edit_g / max(1, len(ref_w))

                        if hyp_b:
                            hyp_b_norm = _norm_for_cer(hyp_b)
                            cer_edit_b = _edit_distance(list(ref_norm), list(hyp_b_norm))
                            cer_edit_sum_b += cer_edit_b
                            sample["cer_beam"] = cer_edit_b / max(1, len(ref_norm))

                            hyp_b_w = _split_for_wer(hyp_b)
                            wer_edit_b = _edit_distance(ref_w, hyp_b_w)
                            wer_edit_sum_b += wer_edit_b
                            sample["wer_beam"] = wer_edit_b / max(1, len(ref_w))

                    fo.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    if n_print < cfg.print_examples:
                        n_print += 1
                        print("\n--- example ---")
                        print("wav:", wav)
                        if ref:
                            print("ref:", ref)
                        print("greedy:", hyp_g)
                        if hyp_b:
                            print("beam  :", hyp_b)
                        print("conf_logp:", confs[i])

    fo.close()

    blank_frac = (blank_cnt / frame_cnt) if frame_cnt > 0 else 0.0
    avg_hyp_len_g = hyp_len_sum_g / max(1, len(ds))
    avg_hyp_len_b = hyp_len_sum_b / max(1, len(ds)) if hyp_len_sum_b > 0 else 0.0

    print("\n========== SUMMARY ==========")
    print(f"output_jsonl: {out_path.resolve()}")
    print(f"blank_frac: {blank_frac*100:.2f}%")
    print(f"avg_hyp_len_greedy: {avg_hyp_len_g:.2f}")
    if avg_hyp_len_b > 0:
        print(f"avg_hyp_len_beam  : {avg_hyp_len_b:.2f}")

    if cer_len_sum > 0:
        cer_g = cer_edit_sum_g / cer_len_sum
        wer_g = wer_edit_sum_g / max(1, wer_len_sum)
        print(f"CER_greedy: {cer_g*100:.2f}%")
        print(f"WER_greedy: {wer_g*100:.2f}%")
        if beam_decoder is not None and beam_decoder.ok:
            cer_b = cer_edit_sum_b / cer_len_sum
            wer_b = wer_edit_sum_b / max(1, wer_len_sum)
            print(f"CER_beam  : {cer_b*100:.2f}%")
            print(f"WER_beam  : {wer_b*100:.2f}%")
    else:
        print("ref 缺失：未计算 CER/WER（test.jsonl 没给 txt/text）")


if __name__ == "__main__":
    main()
