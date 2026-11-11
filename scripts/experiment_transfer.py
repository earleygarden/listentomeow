# -*- coding: utf-8 -*-
"""
实验：迁移学习能否突破情绪分类 56.4% 的天花板。
- 思路：用 AudioSet 预训练音频模型（AST）把每段喵叫编码成语义向量(768维)，
        在其上训一个小分类头，按猫ID分组5折CV。小样本正解。
- 基线：MFCC(268维)+SVM → 0.564；增强版 → 0.553。
- embedding 缓存到 data/，避免重复推理。
- 模型加载失败时自动降级到备选模型（wav2vec2-base）。
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
CACHE_DIR = os.path.join(ROOT, "data", "_emb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CONTEXT_SET = {"B", "F", "I"}
SR = 16000

PRIMARY = "MIT/ast-finetuned-audioset-10-10-0.4593"   # AST 音频集预训练
FALLBACK = "facebook/wav2vec2-base"                   # 兜底：自监督语音特征(768维)


def load_backend(model_id):
    """返回 (feature_extractor, model, is_wav2vec2)。"""
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification, AutoModel
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    is_w2v = False
    try:
        if "wav2vec2" in model_id or "hubert" in model_id:
            model = AutoModel.from_pretrained(model_id)
            is_w2v = True
        else:
            model = ASTForAudioClassification.from_pretrained(model_id)
        return extractor, model, is_w2v
    except Exception as e:
        if model_id != PRIMARY:
            raise
        print(f"[warn] 主模型 {PRIMARY} 加载失败: {e}\n[info] 降级到 {FALLBACK}", flush=True)
        return load_backend(FALLBACK)


def embed_one(extractor, model, is_w2v, y, sr=SR):
    import torch
    if is_w2v:
        inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    else:
        inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last = out.hidden_states[-1]        # (1, T, 768)
        return last.mean(dim=1).squeeze(0).cpu().numpy()


def build_embeddings():
    import torch
    tag = PRIMARY.split("/")[-1]
    emb_path = os.path.join(CACHE_DIR, f"emb_{tag}.npy")
    meta_path = os.path.join(CACHE_DIR, f"meta_{tag}.pkl")
    if os.path.exists(emb_path) and os.path.exists(meta_path):
        print(f"[cache] 复用已缓存的 embedding: {emb_path}", flush=True)
        return np.load(emb_path), pickle.load(open(meta_path, "rb"))

    extractor, model, is_w2v = load_backend(PRIMARY)
    model.eval()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    E, y, groups, names = [], [], [], []
    t0 = time.time()
    ok = 0
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
        try:
            E.append(embed_one(extractor, model, is_w2v, y_sig))
            y.append(ctx); groups.append(cat_id); names.append(base); ok += 1
        except Exception as e:
            print(f"[skip] {base}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(files)} 段, 成功 {ok}, 用时 {time.time()-t0:.0f}s", flush=True)

    E = np.array(E)
    meta = {"y": np.array(y), "groups": np.array(groups), "names": names, "model": tag}
    np.save(emb_path, E)
    pickle.dump(meta, open(meta_path, "wb"))
    print(f"[done] embedding 完成: {E.shape}, 模型={tag}, 用时 {time.time()-t0:.0f}s", flush=True)
    return E, meta


def main():
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    E, meta = build_embeddings()
    y, groups = meta["y"], meta["groups"]
    print(f"\n样本 {len(E)}  维度 {E.shape[1]}  猫数 {len(set(groups))}")

    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    heads = {
        "LogReg(C=1)": Pipeline([("s", StandardScaler()),
                                 ("c", LogisticRegression(C=1, max_iter=2000, class_weight="balanced"))]),
        "LogReg(C=0.1)": Pipeline([("s", StandardScaler()),
                                   ("c", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))]),
        "SVM(rbf)": Pipeline([("s", StandardScaler()),
                              ("c", SVC(kernel="rbf", C=10, gamma="scale",
                                        probability=True, class_weight="balanced", random_state=42))]),
    }
    print("基线对比: MFCC+SVM=0.564 | 增强=0.553\n")
    best = 0.0
    for name, pipe in heads.items():
        y_pred = cross_val_predict(pipe, E, y, groups=groups, cv=gkf)
        acc = accuracy_score(y, y_pred)
        best = max(best, acc)
        print(f"—— {name}: CV准确率 {acc:.3f} ——")
        print(classification_report(y, y_pred, labels=["B", "F", "I"],
                                    target_names=["B满足", "F期待", "I不安"], digits=3, zero_division=0))
        cm = confusion_matrix(y, y_pred, labels=["B", "F", "I"])
        print("混淆矩阵 (行=真实, 列=预测):\n", cm, "\n", flush=True)
    print(f"★ 迁移学习最佳 CV 准确率: {best:.3f}  (原基线 0.564, 提升 {best-0.564:+.3f})")


if __name__ == "__main__":
    main()
