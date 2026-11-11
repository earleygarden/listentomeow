# -*- coding: utf-8 -*-
"""
AST 音频 embedding 提取（torch-free / transformers-free，供 predict.py 推理用）。
- 预处理：复刻官方 ASTFeatureExtractor 的 Kaldi fbank（逐帧去直流+预加重、hann窗、
  frame400/fft512/hop160、|·|²、mel_floor、log、按 AudioSet mean/std 归一化除 std*2）。
  复用 models/ast_feat/ 下由官方 extractor dump 的 mel_filters + window，与官方输出逐元素一致（已验证）。
- 模型：models/ast_embed.onnx（onnxruntime），输入 (1,1024,128)，输出 768 维时间平均隐藏态。
- 设计为惰性加载 + 失败返回 None，让上层可无缝兜底到 MFCC 路径。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FEAT_DIR = os.path.join(ROOT, "models", "ast_feat")
ONNX_PATH = os.path.join(ROOT, "models", "ast_embed.onnx")
SR = 16000
MEAN, STD = -4.2677393, 4.5689974
FRAME, FFT, HOP = 400, 512, 160
MEL_FLOOR = 1.192092955078125e-07   # 2^-23，Kaldi 风格 floor
PREEMPH = 0.97
MAX_LEN = 1024
# 峰值归一化（可选，TMS_PEAK_NORM=1 开启）：训练库 CatMeows 的 peak 中位 0.109，
# 而外部真实录音 0.457（响 4 倍）。AST 用全局 mean/std 归一化，音量系统偏移会直接
# 把 embedding 推出分布 —— 这是队友包 v13 说过的"音量敏感"根因。默认关，实验验证后再决定。
PEAK_NORM = os.environ.get("TMS_PEAK_NORM") == "1"

_session = None
_mel = None
_win = None


def available():
    return (os.path.exists(ONNX_PATH)
            and os.path.exists(os.path.join(FEAT_DIR, "mel_filters.npy"))
            and os.path.exists(os.path.join(FEAT_DIR, "window.npy")))


def _load():
    global _session, _mel, _win
    if _session is not None:
        return
    import onnxruntime as ort
    _mel = np.load(os.path.join(FEAT_DIR, "mel_filters.npy")).astype(np.float64)
    _win = np.load(os.path.join(FEAT_DIR, "window.npy")).astype(np.float64)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session = ort.InferenceSession(ONNX_PATH, so, providers=["CPUExecutionProvider"])


def ast_fbank(y):
    """波形 → (1024,128) 归一化 log-mel（与官方 ASTFeatureExtractor 一致）。"""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < FRAME:
        y = np.pad(y, (0, FRAME - n))
        n = len(y)
    num_frames = 1 + (n - FRAME) // HOP
    # 帧矩阵 (num_frames, 400)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(num_frames)[:, None]
    frames = y[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)      # 逐帧去直流
    pe = frames.copy()
    pe[:, 1:] -= PREEMPH * frames[:, :-1]                     # 逐帧预加重
    pe[:, 0] *= 1 - PREEMPH
    pe *= _win                                                # 加 hann 窗
    buf = np.zeros((num_frames, FFT))
    buf[:, :FRAME] = pe
    spec = np.fft.rfft(buf, axis=1)                           # (num_frames, 257)
    spec = (np.abs(spec) ** 2.0).T                            # (257, num_frames)
    mel = np.maximum(MEL_FLOOR, _mel.T @ spec)                # (128, num_frames)
    fbank = np.log(mel).T                                     # (num_frames, 128)
    m = fbank.shape[0]
    if m < MAX_LEN:
        fbank = np.pad(fbank, ((0, MAX_LEN - m), (0, 0)))
    else:
        fbank = fbank[:MAX_LEN]
    return ((fbank - MEAN) / (STD * 2)).astype(np.float32)    # (1024, 128)


def extract_ast_embedding(y, sr=SR):
    """波形 → 768 维 embedding；任何失败返回 None（上层兜底 MFCC）。"""
    try:
        _load()
        y = np.asarray(y, dtype=np.float64)
        if PEAK_NORM:
            peak = float(np.max(np.abs(y))) if len(y) else 0.0
            if peak > 1e-6:
                y = y * (0.95 / peak)
        x = ast_fbank(y)[None, :, :]                          # (1, 1024, 128)
        emb = _session.run(None, {"input_values": x})[0]      # (1, 768)
        return np.asarray(emb, dtype=np.float32).reshape(-1)
    except Exception:
        return None
