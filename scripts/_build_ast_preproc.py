# -*- coding: utf-8 -*-
"""
构建 torch-free / transformers-free 的 AST 预处理（复刻 Kaldi fbank），
并把官方 extractor 的 mel_filters + window dump 成 npy 复用（消除最难匹配的两处）。
验证：与官方 ASTFeatureExtractor 输出对比，要求相关系数>0.999。
产物：models/ast_feat/mel_filters.npy, window.npy
"""
import os
import sys
import numpy as np
import librosa

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MODEL_DIR = os.path.join(ROOT, "data", "_ast_torch")
FEAT_DIR = os.path.join(ROOT, "models", "ast_feat")
os.makedirs(FEAT_DIR, exist_ok=True)
SR = 16000
MEAN, STD = -4.2677393, 4.5689974
FRAME, FFT, HOP = 400, 512, 160
MEL_FLOOR = 1.192092955078125e-07
PREEMPH = 0.97
MAX_LEN = 1024


def ast_fbank_np(y, mel_filters, window):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < FRAME:
        y = np.pad(y, (0, FRAME - n))
        n = len(y)
    num_frames = 1 + int(np.floor((n - FRAME) / HOP))
    num_freq = FFT // 2 + 1
    spec = np.empty((num_frames, num_freq), dtype=np.complex128)
    buffer = np.zeros(FFT)
    for i in range(num_frames):
        t = i * HOP
        buffer[:FRAME] = y[t:t + FRAME]
        buffer[:FRAME] -= buffer[:FRAME].mean()          # 逐帧去直流
        buffer[1:FRAME] -= PREEMPH * buffer[:FRAME - 1]   # 逐帧预加重
        buffer[0] *= 1 - PREEMPH
        buffer[:FRAME] *= window
        spec[i] = np.fft.rfft(buffer)
    spec = (np.abs(spec) ** 2.0).T                         # (257, num_frames)
    mel = np.maximum(MEL_FLOOR, mel_filters.T @ spec)      # (128, num_frames)
    logmel = np.log(mel)
    fbank = logmel.T                                       # (num_frames, 128)
    m = fbank.shape[0]
    if m < MAX_LEN:
        fbank = np.pad(fbank, ((0, MAX_LEN - m), (0, 0)))
    else:
        fbank = fbank[:MAX_LEN]
    fbank = (fbank - MEAN) / (STD * 2)
    return fbank.astype(np.float32)                        # (1024, 128)


def main():
    from transformers import ASTFeatureExtractor
    ext = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
    mel_filters = np.asarray(ext.mel_filters, dtype=np.float64)
    window = np.asarray(ext.window, dtype=np.float64)
    print("官方 mel_filters:", mel_filters.shape, "| window:", window.shape, flush=True)
    np.save(os.path.join(FEAT_DIR, "mel_filters.npy"), mel_filters)
    np.save(os.path.join(FEAT_DIR, "window.npy"), window)

    # 验证：抽几条不同情境的样本对比
    import glob
    probes = (sorted(glob.glob(os.path.join(ROOT, "samples", "*.wav")))
              + sorted(glob.glob(os.path.join(ROOT, "data", "CatMeows", "dataset", "B_*.wav")))[:1]
              + sorted(glob.glob(os.path.join(ROOT, "data", "CatMeows", "dataset", "I_*.wav")))[:1])
    worst = 1.0
    for f in probes:
        y, _ = librosa.load(f, sr=SR, mono=True)
        official = ext(y, sampling_rate=SR, return_tensors="np")["input_values"][0]  # (1024,128)
        mine = ast_fbank_np(y, mel_filters, window)
        if official.shape != mine.shape:
            print("形状不一致!", official.shape, mine.shape)
            return
        d = np.abs(official - mine).max()
        c = np.corrcoef(official.ravel(), mine.ravel())[0, 1]
        worst = min(worst, c)
        print(f"  {os.path.basename(f):28s} 最大差 {d:.5f}  相关系数 {c:.6f}", flush=True)
    print(f"\n最差相关系数: {worst:.6f}  → {'✅ >0.999 可复用（部署免 torch/transformers）' if worst > 0.999 else '❌ 仍不够，需用官方 extractor'}", flush=True)


if __name__ == "__main__":
    main()
