# -*- coding: utf-8 -*-
"""
实验：在「样本不够」的前提下，用数据增强（滑窗切分 + 变调 + 变速）能否提升猫叫情绪分类。
- 基线：MFCC统计池化(268维) + SVM(RBF)，GroupKFold(按猫ID) → cv_accuracy ≈ 0.564
- 本实验：对同一批音频做增强扩充，分组仍是猫ID（无跨猫泄漏），重测 CV 准确率。
- 目的：不新增录音、不加新依赖，先榨干现有数据的上限。
"""
import os
import sys
import glob
import time
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import extract_features_from_signal, SR  # noqa: E402

DATA_DIR = os.path.join(HERE, "..", "data", "CatMeows", "dataset")
CONTEXT_SET = {"B", "F", "I"}

WIN = 2.0        # 滑窗长度(秒)
HOP = 1.0        # 滑窗步长(秒)
PITCHES = [-2, 2]         # 变调(半音)
STRETCHES = [0.92, 1.08]  # 变速


def variants(y, sr):
    """从一段波形生成增强变体（返回波形列表）。"""
    out = []
    n = len(y)
    w = int(WIN * sr)
    h = int(HOP * sr)
    # 1) 滑窗切分：短片整段，长片切窗
    if n <= w:
        out.append(y.copy())
    else:
        for s in range(0, n - w + 1, h):
            out.append(y[s:s + w])
        if (n - w) % h != 0:
            out.append(y[-w:])
    # 2) 变速 / 变调（对整段各做一次，避免组合爆炸）
    for st in STRETCHES:
        out.append(librosa.effects.time_stretch(y, rate=st))
    for ps in PITCHES:
        out.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=ps))
    return out


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    X, y, groups = [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        base = os.path.basename(f)
        parts = base.split("_")
        ctx, cat_id = parts[0], parts[1]
        if ctx not in CONTEXT_SET:
            continue
        try:
            y_sig, _ = librosa.load(f, sr=SR, mono=True)
        except Exception:
            continue
        for v in variants(y_sig, SR):
            X.append(extract_features_from_signal(v, SR))
            y.append(ctx)
            groups.append(cat_id)
        if (i + 1) % 50 == 0:
            print(f"  ...已处理 {i+1}/{len(files)} 段，累计样本 {len(X)}，用时 {time.time()-t0:.0f}s", flush=True)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    print(f"\n增强后样本数: {len(X)}  维度: {X.shape[1]}  猫数: {len(set(groups))}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("类别分布:", {c: int(n) for c, n in zip(uniq, cnt)})

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, class_weight="balanced", random_state=42)),
    ])
    n_groups = len(set(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=gkf)
    acc = accuracy_score(y, y_pred)
    print(f"\n—— 增强后 分组交叉验证准确率: {acc:.3f}  (基线 0.564) ——\n", flush=True)
    print(classification_report(
        y, y_pred, labels=["B", "F", "I"],
        target_names=["B满足", "F期待", "I不安"], digits=3))
    cm = confusion_matrix(y, y_pred, labels=["B", "F", "I"])
    print("混淆矩阵 (行=真实 B/F/I, 列=预测 B/F/I):")
    print(cm)


if __name__ == "__main__":
    main()
