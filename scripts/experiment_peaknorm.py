# -*- coding: utf-8 -*-
"""
实验：峰值归一化（peak normalization）对情绪分类的影响。

动机（有实测依据，不是拍脑袋）：
    训练库 CatMeows 的 peak 中位 0.109 / RMS 0.0138；外部真实录音 peak 中位 0.457 / RMS 0.0567
    —— 真实音频比训练库**响约 4 倍**。AST 预处理用 AudioSet 的全局 mean/std 归一化，
    音量整体偏移不会归一化掉，会系统性地把 embedding 推出训练分布。
    队友包的 v13 也是这么修的（他们说是"识别不准"的根因）。

数学捷径（已证伪，仅留作记录）：
    理论上峰值归一化 y' = y * 0.95/peak，mel 能量按幅度平方缩放，log-mel 只差常数
    2*ln(0.95/peak)，可以整体平移代替重算。**实测不等价**（最大绝对差 1.09）：
    因为 ast_fbank 里有 MEL_FLOOR=2^-23 的下限截断，而训练库音频本身很小
    （peak 中位 0.109），大量 mel 值触底被截断后在 log 域是常数，平移无法还原。
    所以这里老老实实：先归一化波形，再走完整预处理。

产物：models/cat_emotion_model_ast_peaknorm.joblib（候选，不覆盖正式模型）
用法：python scripts/experiment_peaknorm.py
"""
import os
import sys
import glob
import pickle
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

import joblib  # noqa: E402
import ast_features as af  # noqa: E402
from features import load_audio  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402

CACHE = os.path.join(ROOT, "data", "_emb_cache")
OUT = os.path.join(ROOT, "models", "cat_emotion_model_ast_peaknorm.joblib")


def build_peaknorm_embeddings(names, path_index):
    """先做峰值归一化，再走完整 AST 预处理（不能用平移捷径，见文件头说明）。"""
    af._load()
    out, peaks = [], []
    for i, name in enumerate(names):
        path = path_index.get(name)
        if path is None:
            raise SystemExit("找不到训练文件: %s" % name)
        y = load_audio(path)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        peaks.append(peak if peak > 1e-6 else 1e-6)
        if peak > 1e-6:
            y = y * (0.95 / peak)
        fb = af.ast_fbank(y).astype(np.float32)
        emb = af._session.run(None, {"input_values": fb[None, :, :]})[0]
        out.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        if (i + 1) % 50 == 0:
            print("  ... %d/%d" % (i + 1, len(names)))
    return np.vstack(out), np.array(peaks)


def main():
    meta = pickle.load(open(os.path.join(CACHE, "meta_ast_cls.pkl"), "rb"))
    y, groups, names = meta["y"], meta["groups"], meta["names"]

    path_index = {}
    for pat in ("data/CatMeows/**/*.wav", "data/CatMeows_extras/**/*.wav"):
        for p in glob.glob(os.path.join(ROOT, pat.replace("/", os.sep)), recursive=True):
            path_index[os.path.basename(p)] = p
    missing = [n for n in names if n not in path_index]
    print("训练样本 %d 条，找不到 %d 条" % (len(names), len(missing)))

    print("\n[1/4] 用平移技巧构建峰值归一化 embedding ...")
    E_pn, peaks = build_peaknorm_embeddings(names, path_index)
    print("    完成 %s，peak 中位 %.3f" % (E_pn.shape, float(np.median(peaks))))

    E_orig = np.load(os.path.join(CACHE, "emb_ast_mean.npy"))
    print("\n[2/4] 与原始 embedding 对比：最大差 %.3f、平均差 %.3f（确认输入分布确实变了）"
          % (float(np.abs(E_orig - E_pn).max()), float(np.abs(E_orig - E_pn).mean())))

    print("\n[3/4] 分组交叉验证（按猫 ID，同一只猫不会跨训练/测试）...")
    results = {}
    for tag, E in (("原始(现状)", E_orig), ("峰值归一化", E_pn)):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        probability=True, class_weight="balanced", random_state=42)),
        ])
        gkf = GroupKFold(n_splits=min(5, len(set(groups))))
        pred = cross_val_predict(pipe, E, y, groups=groups, cv=gkf)
        acc = accuracy_score(y, pred)
        results[tag] = {"acc": acc, "E": E}
        print("\n  —— %s：CV = %.3f ——" % (tag, acc))
        print(classification_report(y, pred, labels=["B", "F", "I"],
                                    target_names=["B满足", "F期待", "I不安"], digits=3))

    print("\n[4/4] 训练并保存候选模型（峰值归一化版）...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(E_pn, y)
    joblib.dump({
        "pipeline": pipe,
        "classes": list(pipe.classes_),
        "feature_dim": int(E_pn.shape[1]),
        "feature_type": "ast_embedding_768_meanpool_peaknorm",
        "cv_accuracy": float(results["峰值归一化"]["acc"]),
        "peak_norm": True,
        "note": "AST 768维 + 峰值归一化（y*0.95/peak）重训；实验对照 CV %.3f → %.3f"
                % (results["原始(现状)"]["acc"], results["峰值归一化"]["acc"]),
    }, OUT)
    np.save(os.path.join(CACHE, "emb_ast_mean_peaknorm.npy"), E_pn)
    print("    候选模型: %s" % OUT)
    print("    说明：CV 高低不是唯一标准，还要看库外真实音频 —— 用"
          "\n      TMS_PEAK_NORM=1 TMS_AST_MODEL=models/cat_emotion_model_ast_peaknorm.joblib"
          "\n      python scripts/bench_external.py 对比 A 组准确率。")


if __name__ == "__main__":
    main()
