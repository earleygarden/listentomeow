# -*- coding: utf-8 -*-
"""
特征/模型对比实验：目标是在「按猫分组」的严格评估下尽量拉高准确率。
- 特征 v2：MFCC+delta+delta2 统计(240) + 谱质心/带宽/rolloff/flatness/ZCR/RMS/f0 统计
- 模型对比：SVM / RandomForest / GradientBoosting / ExtraTrees / LogisticRegression
"""
import os
import sys
import glob
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "CatMeows", "dataset")
SR = 16000
N_MFCC = 20
N_FFT = 2048
HOP = 512


def _stats(m):
    """一维或多维特征沿时间轴的 4 项统计。"""
    m = np.atleast_2d(m)
    return np.concatenate([m.mean(1), m.std(1), m.min(1), m.max(1)])


def features_v2(y, sr=SR):
    y = librosa.effects.trim(y, top_db=30)[0]
    min_len = int(0.5 * sr)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                hop_length=HOP, fmin=50, fmax=sr // 2 - 1)
    d1 = librosa.feature.delta(mfcc, width=3)
    d2 = librosa.feature.delta(mfcc, order=2, width=3)

    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    flat = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)

    # 基频 f0（yin），过滤静音帧
    try:
        f0 = librosa.yin(y, fmin=100, fmax=1500, sr=sr, frame_length=N_FFT * 2)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        f0st = np.array([f0.mean(), f0.std(), f0.min(), f0.max()]) if len(f0) else np.zeros(4)
    except Exception:
        f0st = np.zeros(4)

    return np.concatenate([
        _stats(mfcc), _stats(d1), _stats(d2),
        _stats(cent), _stats(bw), _stats(roll), _stats(flat), _stats(zcr), _stats(rms),
        f0st.astype(np.float32),
    ]).astype(np.float32)


def load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    X, y, g = [], [], []
    for f in files:
        ctx = os.path.basename(f).split("_")[0]
        if ctx not in ("B", "F", "I"):
            continue
        sig, _ = librosa.load(f, sr=SR, mono=True)
        X.append(features_v2(sig))
        y.append(ctx)
        g.append(os.path.basename(f).split("_")[1])
    return np.array(X), np.array(y), np.array(g)


def main():
    X, y, g = load()
    print(f"样本 {len(X)}  维度 {X.shape[1]}  分组 {len(set(g))}")

    models = {
        "SVM(rbf,C=10)": Pipeline([("s", StandardScaler()), ("m", SVC(C=10, gamma="scale", class_weight="balanced"))]),
        "SVM(rbf,C=30)": Pipeline([("s", StandardScaler()), ("m", SVC(C=30, gamma="scale", class_weight="balanced"))]),
        "RandomForest": Pipeline([("s", StandardScaler()), ("m", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42))]),
        "ExtraTrees": Pipeline([("s", StandardScaler()), ("m", ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=42))]),
        "GradBoosting": Pipeline([("s", StandardScaler()), ("m", GradientBoostingClassifier(random_state=42))]),
        "LogReg": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    }

    gkf = GroupKFold(n_splits=5)
    print("\n模型对比（按猫分组5折）:")
    best, best_acc = None, -1
    for name, m in models.items():
        pred = cross_val_predict(m, X, y, groups=g, cv=gkf)
        acc = accuracy_score(y, pred)
        flag = ""
        if acc > best_acc:
            best_acc, best = acc, name
            flag = "  <== best"
        print(f"  {name:16s} acc={acc:.3f}{flag}")
    print(f"\n最优: {best}  acc={best_acc:.3f}")

    # 特征重要性（用 ExtraTrees 看哪些特征组有用）
    et = ExtraTreesClassifier(n_estimators=400, random_state=42).fit(X, y)
    imp = et.feature_importances_
    print(f"\n特征维度总数 {len(imp)}; 前10重要维度: {np.argsort(imp)[::-1][:10].tolist()}")


if __name__ == "__main__":
    main()
