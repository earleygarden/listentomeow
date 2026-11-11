# -*- coding: utf-8 -*-
"""验证 predict.py 该用哪种 AST 预处理：
1) 官方 ASTFeatureExtractor(return_tensors=np) 是否 torch-free；
2) 手工 librosa mel 能否对上官方（含正确的 (1024,128) 布局）；
3) 两者喂导出 ONNX 的 embedding 是否一致。
"""
import os
import sys
import numpy as np
import librosa

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MODEL_DIR = os.path.join(ROOT, "data", "_ast_torch")
ONNX = os.path.join(ROOT, "models", "ast_embed.onnx")
SAMPLE = os.path.join(ROOT, "samples", "sample_饿_等食.wav")
SR = 16000
MEAN, STD = -4.2677393, 4.5689974

# 1) 官方 extractor 是否 torch-free
import transformers  # noqa
from transformers import ASTFeatureExtractor  # noqa
torch_loaded_before = "torch" in sys.modules
ext = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
y, _ = librosa.load(SAMPLE, sr=SR, mono=True)
inp = ext(y, sampling_rate=SR, return_tensors="np")
torch_loaded_after = "torch" in sys.modules
official = inp["input_values"]
print("torch 已加载(import transformers前):", torch_loaded_before)
print("torch 已加载(跑完 extractor 后):", torch_loaded_after, "  ← False=预处理不依赖 torch")
print("官方 input_values 形状:", official.shape)


# 2) 手工 mel → (1024,128)
def manual_ast(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160,
                                         n_mels=128, center=True, power=2.0,
                                         fmin=0.0, fmax=sr // 2)
    logmel = np.log(mel + 1e-6)
    logmel = (logmel - MEAN) / STD
    T = logmel.shape[1]
    if T < 1024:
        logmel = np.pad(logmel, ((0, 0), (0, 1024 - T)))
    else:
        logmel = logmel[:, :1024]
    return logmel.T.astype(np.float32)   # (1024, 128) = (时间, mel)

manual = manual_ast(y, SR)[None, :, :]    # (1, 1024, 128)
print("\n手工 input 形状:", manual.shape, "| 官方:", official.shape)
if manual.shape == official.shape:
    diff = np.abs(manual - official)
    corr = np.corrcoef(manual.ravel(), official.ravel())[0, 1]
    print(f"手工 vs 官方：最大差 {diff.max():.4f}  平均差 {diff.mean():.4f}  相关系数 {corr:.5f}")

# 3) 两者喂 ONNX 比 embedding
import onnxruntime as ort
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
e_off = sess.run(None, {"input_values": official})[0]
e_man = sess.run(None, {"input_values": manual})[0]
if manual.shape == official.shape:
    d = np.abs(e_off - e_man)
    c = np.corrcoef(e_off.ravel(), e_man.ravel())[0, 1]
    print(f"\nONNX embedding：官方 vs 手工 最大差 {d.max():.4f}  相关系数 {c:.5f}")
    print("（相关系数>0.999 且差很小 → 手工预处理可用，predict.py 可免 transformers 依赖）")
else:
    print("\n形状不一致，无法直接比 embedding")
