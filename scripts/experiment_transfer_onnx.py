# -*- coding: utf-8 -*-
"""
实验：迁移学习(AST/ONNX)能否突破情绪分类 56.4% 天花板。onnxruntime 路线，无需 torch。
- 预处理复刻 ASTFeatureExtractor：16kHz, 128 mel bins, n_fft=400, hop=160,
  log(mel+1e-6) 后按 AudioSet 的 mean/std 归一化，时间补齐/截断到 1024 帧。
- 安全闸：先跑几只猫叫，确认 AudioSet 的 Cat/Meow/Purr 等类被激活，证明预处理正确。
- 输出取 768 维 CLS embedding（若模型只吐 527 类 logits，则退用 logits 作特征）。
- 按猫ID分组5折CV，与 MFCC+SVM(0.564)/增强(0.553) 对比。
"""
import os
import sys
import glob
import json
import time
import pickle
import numpy as np
import librosa

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA_DIR = os.path.join(ROOT, "data", "CatMeows", "dataset")
AST_DIR = os.path.join(ROOT, "data", "_ast_onnx")
CACHE_DIR = os.path.join(ROOT, "data", "_emb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CONTEXT_SET = {"B", "F", "I"}
SR = 16000
N_FFT, HOP, N_MELS, MAX_LEN = 400, 160, 128, 1024
MEAN, STD = -4.2677393, 4.5689974


def ast_preprocess(waveform, sr, norm_variant=1):
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, center=True, power=2.0, fmin=0.0, fmax=sr // 2)
    logmel = np.log(mel + 1e-6)
    if norm_variant == 1:
        logmel = (logmel - MEAN) / STD
    else:
        logmel = (logmel - MEAN) / (STD * 2)
    T = logmel.shape[1]
    if T < MAX_LEN:
        logmel = np.pad(logmel, ((0, 0), (0, MAX_LEN - T)), mode="constant")
    else:
        logmel = logmel[:, :MAX_LEN]
    return logmel.astype(np.float32)  # (128, 1024)


def load_session():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(AST_DIR, "model.onnx"), so,
                                providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
    return sess, ins, outs


def run_model(sess, input_name, logmel):
    import onnxruntime as ort
    x = logmel[None, :, :]                      # (1, 128, 1024)
    try:
        return sess.run(None, {input_name: x})
    except Exception:
        return sess.run(None, {input_name: x.transpose(0, 2, 1)})  # 若模型要 (1,1024,128)


def embedding_from_outputs(outputs):
    """从 ONNX 输出里取向量：优先 768 维隐藏态，退用 logits。"""
    for o in outputs:
        a = np.asarray(o)
        if a.ndim == 3:            # (1, T, hidden) → 时间平均
            return a.mean(axis=1).reshape(-1)
    # 只剩 (1, C) logits
    for o in outputs:
        a = np.asarray(o)
        if a.ndim == 2:
            return a.reshape(-1)
    return np.asarray(outputs[0]).reshape(-1)


def validate(sess, input_name, outs, id2label):
    print("\n=== 安全闸：验证预处理是否正确（猫叫应激活 Cat/Meow/Purr 类）===", flush=True)
    probes = sorted(glob.glob(os.path.join(DATA_DIR, "B_*.wav")))[:2] + \
             sorted(glob.glob(os.path.join(DATA_DIR, "I_*.wav")))[:1]
    for f in probes:
        y, _ = librosa.load(f, sr=SR, mono=True)
        logmel = ast_preprocess(y, SR, norm_variant=1)
        outputs = run_model(sess, input_name, logmel)
        logits = None
        for o in outputs:
            a = np.asarray(o)
            if a.ndim == 2 and a.shape[-1] >= 500:
                logits = a.reshape(-1)
                break
        if logits is None:
            print(f"  {os.path.basename(f)}: 模型无 logits 输出(只吐隐藏态)，跳过类别验证", flush=True)
            return
        top = np.argsort(logits)[::-1][:6]
        names = [(id2label.get(str(int(i)), "?"), round(float(logits[i]), 2)) for i in top]
        print(f"  {os.path.basename(f)} top6: {names}", flush=True)
    print("  → 若上面出现 Cat/Meow/Purr/Animal 之类，说明预处理正确。\n", flush=True)


def build_embeddings(sess, input_name, outs):
    tag = "ast_onnx"
    emb_path = os.path.join(CACHE_DIR, f"emb_{tag}.npy")
    meta_path = os.path.join(CACHE_DIR, f"meta_{tag}.pkl")
    if os.path.exists(emb_path) and os.path.exists(meta_path):
        print(f"[cache] 复用 embedding: {emb_path}", flush=True)
        return np.load(emb_path), pickle.load(open(meta_path, "rb"))
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    E, y, groups, names = [], [], [], []
    t0 = time.time(); ok = 0
    for i, f in enumerate(files):
        base = os.path.basename(f); parts = base.split("_")
        ctx, cat_id = parts[0], parts[1]
        if ctx not in CONTEXT_SET:
            continue
        try:
            y_sig, _ = librosa.load(f, sr=SR, mono=True)
        except Exception:
            continue
        try:
            logmel = ast_preprocess(y_sig, SR, norm_variant=1)
            outputs = run_model(sess, input_name, logmel)
            E.append(embedding_from_outputs(outputs))
            y.append(ctx); groups.append(cat_id); names.append(base); ok += 1
        except Exception as e:
            print(f"[skip] {base}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(files)} 段, 成功 {ok}, 用时 {time.time()-t0:.0f}s", flush=True)
    E = np.array(E)
    meta = {"y": np.array(y), "groups": np.array(groups), "names": names, "model": tag}
    np.save(emb_path, E)
    pickle.dump(meta, open(meta_path, "wb"))
    print(f"[done] embedding: {E.shape}, 用时 {time.time()-t0:.0f}s", flush=True)
    return E, meta


def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    sess, ins, outs = load_session()
    print("ONNX 输入:", ins)
    print("ONNX 输出:", outs, flush=True)
    input_name = ins[0][0]
    id2label = json.load(open(os.path.join(AST_DIR, "config.json"))).get("id2label", {})
    validate(sess, input_name, outs, id2label)

    E, meta = build_embeddings(sess, input_name, outs)
    y, groups = meta["y"], meta["groups"]
    print(f"\n样本 {len(E)}  维度 {E.shape[1]}  猫数 {len(set(groups))}")

    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    heads = {
        "LogReg(C=1)": Pipeline([("s", StandardScaler()),
                                 ("c", LogisticRegression(C=1, max_iter=3000, class_weight="balanced"))]),
        "LogReg(C=0.1)": Pipeline([("s", StandardScaler()),
                                   ("c", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))]),
        "SVM(rbf)": Pipeline([("s", StandardScaler()),
                              ("c", SVC(kernel="rbf", C=10, gamma="scale",
                                        probability=True, class_weight="balanced", random_state=42))]),
    }
    print("基线: MFCC+SVM=0.564 | 增强=0.553\n")
    best = 0.0
    for name, pipe in heads.items():
        y_pred = cross_val_predict(pipe, E, y, groups=groups, cv=gkf)
        acc = accuracy_score(y, y_pred)
        best = max(best, acc)
        print(f"—— {name}: CV准确率 {acc:.3f} ——")
        print(classification_report(y, y_pred, labels=["B", "F", "I"],
                                    target_names=["B满足", "F期待", "I不安"], digits=3, zero_division=0))
        print("混淆矩阵 (行=真实, 列=预测):\n", confusion_matrix(y, y_pred, labels=["B", "F", "I"]), "\n", flush=True)
    print(f"★ 迁移学习最佳 CV: {best:.3f}  (基线 0.564, 提升 {best-0.564:+.3f})")


if __name__ == "__main__":
    main()
