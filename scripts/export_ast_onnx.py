# -*- coding: utf-8 -*-
"""
导出 AST embedding 为 ONNX：输入 AST 的 mel 频谱 (batch,128,1024)，
输出 768 维时间平均隐藏态（分类头之前的语义向量）。
产物：models/ast_embed.onnx（供 predict.py 用 onnxruntime 推理，运行时无需 torch）。
"""
import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MODEL_DIR = os.path.join(ROOT, "data", "_ast_torch")
OUT = os.path.join(ROOT, "models", "ast_embed.onnx")

import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor  # noqa: E402
import librosa  # noqa: E402

SR = 16000


def confirm_input_shape(extractor):
    """用官方 feature extractor 跑一段真音频，确认 input_values 形状。"""
    f = os.path.join(ROOT, "samples", "sample_饿_等食.wav")
    y, _ = librosa.load(f, sr=SR, mono=True)
    inp = extractor(y, sampling_rate=SR, return_tensors="pt")
    print("官方 extractor 输入形状:", {k: tuple(v.shape) for k, v in inp.items()}, flush=True)
    return tuple(inp["input_values"].shape)


class ASTEmbedder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_values):
        out = self.m(input_values=input_values, output_hidden_states=True)
        last = out.hidden_states[-1]          # (B, T, 768)
        return last.mean(dim=1)               # (B, 768)


def main():
    extractor = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
    shape = confirm_input_shape(extractor)     # e.g. (1, 128, 1024)
    model = ASTForAudioClassification.from_pretrained(MODEL_DIR).eval()

    emb = ASTEmbedder(model).eval()
    dummy = torch.randn(*shape)
    with torch.no_grad():
        ref = emb(dummy).numpy()

    print(f"导出 ONNX → {OUT} (输入 {shape})", flush=True)
    try:
        torch.onnx.export(
            emb, (dummy,), OUT,
            input_names=["input_values"], output_names=["embedding"],
            opset_version=14,
            dynamic_axes={"input_values": {0: "batch"}, "embedding": {0: "batch"}},
            dynamo=False,
        )
    except TypeError:
        # 老/新 torch 不支持 dynamo 参数时的兼容
        torch.onnx.export(
            emb, (dummy,), OUT,
            input_names=["input_values"], output_names=["embedding"],
            opset_version=14,
            dynamic_axes={"input_values": {0: "batch"}, "embedding": {0: "batch"}},
        )

    # 用 onnxruntime 验证：输出形状与数值应和 torch 一致
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input_values": dummy.numpy()})[0]
    print("torch 输出:", ref.shape, "| onnx 输出:", got.shape, flush=True)
    print("最大数值差:", float(np.abs(ref - got).max()), flush=True)
    print("模型大小: %.1f MB" % (os.path.getsize(OUT) / 1048576), flush=True)
    assert ref.shape == got.shape and np.abs(ref - got).max() < 1e-3, "ONNX 与 torch 输出不一致！"
    print("✅ 导出并验证通过", flush=True)


if __name__ == "__main__":
    main()
