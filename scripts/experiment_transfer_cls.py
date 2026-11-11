# -*- coding: utf-8 -*-
"""
实验：AST 迁移学习的"公平测试"——用分类头【之前】的 768 维 CLS/隐藏态向量，而非 527 类 logits。
- 从本地目录 data/_ast_torch/ 加载官方权重 + 官方 ASTFeatureExtractor（预处理 100% 正确）。
- 抽两种 embedding：CLS token(768) 与 时间平均(768)。
- 按猫ID分组5折CV，与 MFCC+SVM(0.564)/增强(0.553)/ONNX-logits(0.525) 对比。
"""
import os
import sys
import glob
import time
import pickle
import numpy as np
import librosa

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA_DIR = os.path.join(ROOT, "data", "CatMeows", "dataset")
MODEL_DIR = os.path.join(ROOT, "data", "_ast_torch")
CACHE_DIR = os.path.join(ROOT, "data", "_emb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CONTEXT_SET = {"B", "F", "I"}
SR = 16000


def load_backend():
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    extractor = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
    model = ASTForAudioClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return extractor, model


def embed_one(extractor, model, y):
    import torch
    inputs = extractor(y, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    last = out.hidden_states[-1][0]          # (T, 768)
    return last[0].numpy(), last.mean(0).numpy()   # CLS token, mean-pool


def build():
    extractor, model = load_backend()
    # 先验证：logits 里 Cat/Meow 是否激活（官方预处理自检）
    import json
    id2label = json.load(open(os.path.join(MODEL_DIR, "config.json"))).get("id2label", {})
    probes = sorted(glob.glob(os.path.join(DATA_DIR, "B_*.wav")))[:1] + \
             sorted(glob.glob(os.path.join(DATA_DIR, "I_*.wav")))[:1]
    print("=== 官方管线自检（猫叫应激活 Cat/Meow）===", flush=True)
    for f in probes:
        y, _ = librosa.load(f, sr=SR, mono=True)
        import torch
        inp = extractor(y, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inp).logits[0].numpy()
        top = np.argsort(logits)[::-1][:5]
        print(" ", os.path.basename(f), [(id2label.get(str(int(i)), "?"), round(float(logits[i]), 2)) for i in top], flush=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    CLS, MEAN, y, groups, names = [], [], [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        base = os.path.basename(f); parts = base.split("_")
        ctx, cat_id = parts[0], parts[1]
        if ctx not in CONTEXT_SET:
            continue
        try:
            y_sig, _ = librosa.load(f, sr=SR, mono=True)
            c, m = embed_one(extractor, model, y_sig)
            CLS.append(c); MEAN.append(m); y.append(ctx); groups.append(cat_id); names.append(base)
        except Exception as e:
            print(f"[skip] {base}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(files)} 段, 用时 {time.time()-t0:.0f}s", flush=True)
    meta = {"y": np.array(y), "groups": np.array(groups), "names": names}
    np.save(os.path.join(CACHE_DIR, "emb_ast_cls.npy"), np.array(CLS))
    np.save(os.path.join(CACHE_DIR, "emb_ast_mean.npy"), np.array(MEAN))
    pickle.dump(meta, open(os.path.join(CACHE_DIR, "meta_ast_cls.pkl"), "wb"))
    print(f"[done] CLS {np.array(CLS).shape} 用时 {time.time()-t0:.0f}s", flush=True)
    return np.array(CLS), np.array(MEAN), meta


def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    cache_cls = os.path.join(CACHE_DIR, "emb_ast_cls.npy")
    cache_meta = os.path.join(CACHE_DIR, "meta_ast_cls.pkl")
    if not (os.path.exists(cache_cls) and os.path.exists(cache_meta)):
        build()
    CLS = np.load(os.path.join(CACHE_DIR, "emb_ast_cls.npy"))
    MEAN = np.load(os.path.join(CACHE_DIR, "emb_ast_mean.npy"))
    meta = pickle.load(open(os.path.join(CACHE_DIR, "meta_ast_cls.pkl"), "rb"))
    y, groups = meta["y"], meta["groups"]
    print(f"\n样本 {len(CLS)}  维度 {CLS.shape[1]}  猫数 {len(set(groups))}")
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))

    heads = {
        "LogReg(C=1)": Pipeline([("s", StandardScaler()),
                                 ("c", LogisticRegression(C=1, max_iter=3000, class_weight="balanced"))]),
        "SVM(rbf)": Pipeline([("s", StandardScaler()),
                              ("c", SVC(kernel="rbf", C=10, gamma="scale",
                                        probability=True, class_weight="balanced", random_state=42))]),
    }
    print("基线: MFCC+SVM=0.564 | 增强=0.553 | ONNX-logits=0.525\n")
    for emb_name, E in [("CLS token(768)", CLS), ("时间平均(768)", MEAN)]:
        best = 0.0
        for hname, pipe in heads.items():
            y_pred = cross_val_predict(pipe, E, y, groups=groups, cv=gkf)
            acc = accuracy_score(y, y_pred)
            best = max(best, acc)
            print(f"[{emb_name} | {hname}] CV准确率 {acc:.3f}", flush=True)
        # 打印最佳那个的详情
        y_pred = cross_val_predict(heads["LogReg(C=1)"], E, y, groups=groups, cv=gkf)
        print(classification_report(y, y_pred, labels=["B", "F", "I"],
                                    target_names=["B满足", "F期待", "I不安"], digits=3, zero_division=0))
        print("混淆矩阵:\n", confusion_matrix(y, y_pred, labels=["B", "F", "I"]), "\n", flush=True)
        print(f"  → {emb_name} 最佳: {best:.3f}\n", flush=True)


if __name__ == "__main__":
    main()
