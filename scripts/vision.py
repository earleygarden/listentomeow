# -*- coding: utf-8 -*-
"""
视觉分析脚本：视频/音频文件 → 抽取若干帧 → 猫检测(DETR-resnet-50 ONNX, CPU) → 画面状态摘要

用法:
    python vision.py <媒体文件>              # 文本输出
    python vision.py <媒体文件> --json       # JSON 输出（供本地服务调用）
    python vision.py <媒体文件> --json -n 12 # 指定抽帧数

设计原则（重要，勿擅自放宽）:
    只报告"能从画面上量出来的东西"：
      L0 画面里有没有猫            —— 目标检测，可靠
      L1 猫在哪、多大、在不在动     —— 几何与光流，中等可靠
    绝对不做、也不暗示：
      L2 猫的姿势含义（伸懒腰=舒服？）  —— 做不到
      L3 猫的情绪（高兴/害怕）          —— 做不到，必须由声音判断
    视觉只提供"当时画面里发生了什么"，不参与情绪结论。这是本项目"诚实边界"要求的一部分。

返回结构（--json，单行 JSON）:
    非视频（纯音频）:
        { ok:true, media:"audio", has_cat:null, note:"这是纯音频…" }
    视频:
        { ok:true, media:"video", duration_sec, frames:{sampled,with_cat},
          has_cat:true, cat:{appear_ratio,size_ratio,size_label,position,stability,best_conf},
          scene:{multi_cat,max_cats,multi_frame_ratio},
          motion:{level,cat_motion_ps,frame_motion_ps,camera_moving},
          note }
    出错（不抛异常，返回 ok:false，调用方据此降级）:
        { ok:false, media:"unknown", error:"…", note }
"""
import os
import re
import sys
import json
import glob
import shutil
import tempfile
import argparse
import subprocess

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "models", "detr_cat_quant.onnx")

CAT_ID = 17          # COCO 91 类体系中 cat 的 id（DETR facebook/detr-resnet-50 用的就是这套）
MAX_SEC = 30.0       # 与 features.py 的 MAX_SEC 对齐：音画分析覆盖同一时间窗
SCAN_LONG = 448      # 第一遍扫描：长边缩到 448，实测单帧约 200ms
ZOOM_LONG = 800      # 第一遍全灭时的补扫分辨率
CONF_HIT = 0.5       # 判定"这一帧里有猫"的置信度
CONF_RETRY = 0.25    # 补扫时的放宽阈值

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".amr", ".wma"}

# ImageNet 归一化（DETR 官方预处理）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- ffmpeg / 探测

def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path):
    """从 ffmpeg -i 的 stderr 里读时长；读不到返回 None。"""
    try:
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path],
                           capture_output=True, timeout=60)
        s = r.stderr.decode("utf-8", "ignore")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", s)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        # 直播流/无时长信息
        return None
    except Exception:
        return None


def has_video_stream(path):
    try:
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path],
                           capture_output=True, timeout=60)
        return bool(re.search(r"Stream #\d+:\d+.*: Video:", r.stderr.decode("utf-8", "ignore")))
    except Exception:
        return False


# ---------------------------------------------------------------- 抽帧

def pick_frame_count(dur, force=None):
    if force:
        return max(2, min(int(force), 24))
    if dur is None:
        return 8
    if dur <= 4:
        return 6
    if dur <= 10:
        return 8
    if dur <= 20:
        return 10
    return 12


def extract_frames(path, dur, n, long_side, outdir):
    """均匀抽 n 帧，长边缩到 long_side。返回按时间排序的图片路径列表。"""
    span = min(dur, MAX_SEC) if dur else MAX_SEC
    step = max(span / n, 0.04)
    pattern = os.path.join(outdir, "f_%03d.jpg")
    cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-i", path,
           "-vf", "fps=%.4f,scale=%d:%d:force_original_aspect_ratio=decrease"
                  % (1.0 / step, long_side, long_side),
           "-frames:v", str(n), "-q:v", "3", pattern]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return [], step
    if r.returncode != 0:
        return [], step
    return sorted(glob.glob(os.path.join(outdir, "f_*.jpg"))), step


# ---------------------------------------------------------------- 检测

def _make_session(threads=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads or min(8, os.cpu_count() or 4)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])


def _prep_bgr(bgr):
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)


def detect_cats(sess, bgr, conf=CONF_HIT):
    """返回 [(conf, (x1,y1,x2,y2)), ...]，坐标为像素（相对当前帧），按置信度降序。"""
    px = _prep_bgr(bgr)
    h, w = bgr.shape[:2]
    mask = np.ones((1, 64, 64), dtype=np.int64)
    logits, boxes = sess.run(None, {"pixel_values": px, "pixel_mask": mask})
    lg = np.asarray(logits)[0]
    bx = np.asarray(boxes)[0]
    e = np.exp(lg - lg.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    sc = p[:, :-1]                              # 去掉 no-object 类
    cls = sc.argmax(-1)
    scr = sc.max(-1)
    out = []
    for q in range(len(scr)):
        if int(cls[q]) == CAT_ID and scr[q] >= conf:
            cx, cy, bw, bh = [float(v) for v in bx[q][:4]]
            x1 = max(0.0, (cx - bw / 2) * w); y1 = max(0.0, (cy - bh / 2) * h)
            x2 = min(float(w), (cx + bw / 2) * w); y2 = min(float(h), (cy + bh / 2) * h)
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                out.append((float(scr[q]), (x1, y1, x2, y2)))
    return sorted(out, reverse=True)


def _iou(a, b):
    """两个 (x1,y1,x2,y2) 框的交并比 —— 用于同一只猫被检出多个框时的去重计数。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def count_distinct(boxes, iou_th=0.45):
    """去掉互相高度重叠的框之后，还剩几只不同的猫。"""
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_th for k in kept):
            kept.append(b)
    return len(kept)


# ---------------------------------------------------------------- 几何 / 运动

_POS = (("左上", "上中", "右上"), ("左中", "中央", "右中"), ("左下", "下中", "右下"))


def position_label(box, w, h):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / max(w, 1)
    cy = ((y1 + y2) / 2) / max(h, 1)
    col = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
    row = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
    return _POS[row][col]


def size_label(ratio):
    if ratio < 0.06:
        return "离得比较远"
    if ratio < 0.18:
        return "中等大小"
    if ratio < 0.40:
        return "占了不小的一块"
    return "几乎占满画面"


def flow_magnitude(g1, g2, box=None):
    """Farneback 光流平均幅度（像素/帧）。box 给定时只统计该区域。"""
    import cv2
    f = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
    if box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(mag.shape[1], max(x2, x1 + 1)); y2 = min(mag.shape[0], max(y2, y1 + 1))
        sub = mag[y1:y2, x1:x2]
        if sub.size < 32:
            return None
        return float(np.mean(sub))
    return float(np.mean(mag))


def _thumb(bgr, box, long_side=320, quality=68):
    """把检到猫的那一帧画上检测框，缩成小图返回 data URI —— 用户可自行核对，比任何文字都诚实。"""
    import base64
    import cv2
    img = bgr.copy()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
    cv2.rectangle(img, (x1, y1), (x2, y2), (94, 165, 216), 2)      # 品牌金 #D8A55E 的 BGR 写法
    tag = "cat"                                                    # 用 Hershey 字体，避开中文字形缺失
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    ty = y1 - 5
    if ty - th - 4 < 0:                                            # 框贴顶时角标改画在框内
        ty = y1 + th + 8
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 8, ty + 2), (94, 165, 216), -1)
    cv2.putText(img, tag, (x1 + 4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    k = long_side / float(max(h, w))
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------- 主流程

def _r3(v):
    return None if v is None else round(float(v), 3)


def analyze(path, force_n=None, as_json=False):
    ext = os.path.splitext(path)[1].lower()
    base = {
        "ok": True,
        "media": "audio" if ext in AUDIO_EXT else ("video" if ext in VIDEO_EXT else "unknown"),
        "has_cat": None,
        "note": "",
    }

    if not os.path.exists(path):
        base.update(ok=False, media="unknown", error="文件不存在")
        return _emit(base, as_json)

    if not os.path.exists(MODEL_PATH):
        base.update(ok=False, media=base["media"],
                    error="猫检测模型缺失（models/detr_cat_quant.onnx）")
        return _emit(base, as_json)

    is_video = has_video_stream(path)
    if not is_video:
        # ffmpeg 完全读不出时长 → 文件坏了或不是媒体，别假装成"纯音频"
        if probe_duration(path) is None:
            base.update(ok=False, media="unknown",
                        error="这个文件读不出来（可能已损坏，或者不是音频/视频文件）",
                        note="画面没读出来，我只听了声音。")
            return _emit(base, as_json)
        # 确属纯音频：没有画面可看，如实说明
        base.update(media="audio", has_cat=None,
                    note="这是纯音频，没有画面可看 —— 我只听了声音。")
        return _emit(base, as_json)

    base["media"] = "video"
    dur = probe_duration(path)
    n = pick_frame_count(dur, force_n)
    work = tempfile.mkdtemp(prefix="tm_vis_")
    try:
        frames, step = extract_frames(path, dur, n, SCAN_LONG, work)
        if not frames:
            base.update(ok=False, media="video", error="视频解码失败（可能是损坏文件或罕见编码）",
                        note="画面没读出来，我只听了声音。")
            return _emit(base, as_json)

        sess = _make_session()
        import cv2
        imgs = []
        # 每帧: (best_conf | None, box | None, (w,h), imgs 下标)
        per_frame = []
        strong_counts = []                      # 每帧去重后的猫数（>=2 即"这帧不止一只猫"）
        weak_any = False
        for fp in frames:
            bgr = cv2.imread(fp)
            if bgr is None:
                continue
            idx = len(imgs)
            imgs.append(bgr)
            h, w = bgr.shape[:2]
            # 一次推理取回所有候选；CONF_HIT 以上才算"这帧有猫"，弱候选只用来触发补扫
            raw = detect_cats(sess, bgr, CONF_RETRY)
            strong = [x for x in raw if x[0] >= CONF_HIT]
            weak_any = weak_any or bool(raw)
            strong_counts.append(count_distinct([x[1] for x in strong]))
            if strong:
                per_frame.append((strong[0][0], strong[0][1], (w, h), idx))
            else:
                per_frame.append((None, None, (w, h), idx))

        with_cat = [x for x in per_frame if x[0] is not None]

        # 一遍全灭 → 用中间几帧放大重扫一次，兜"猫很小/很远"（放大后仍需过 CONF_HIT 才算数）
        if not with_cat and weak_any and imgs:
            mid = len(imgs) // 2
            cand = [mid] + ([0, len(imgs) - 1] if len(imgs) > 2 else [])
            for ci in cand:
                bgr = imgs[ci]
                k = ZOOM_LONG / float(max(bgr.shape[:2]))
                up = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
                hits = [x for x in detect_cats(sess, up, CONF_RETRY) if x[0] >= CONF_HIT]
                if hits:
                    b = hits[0][1]
                    # 放大是等比，坐标除以 k 即回到扫描尺度
                    box = (b[0] / k, b[1] / k, b[2] / k, b[3] / k)
                    with_cat = [(hits[0][0], box, (bgr.shape[1], bgr.shape[0]), ci)]
                    break

        # --- 场景统计：画面里到底是几只猫（去重后）---
        # 多猫同框时声音往往是"合唱"，情绪模型分不清是谁在叫 —— 这个事实必须让上层知道
        n_sampled = max(len(per_frame), 1)
        max_cats = max(strong_counts) if strong_counts else 0
        multi_ratio = (sum(1 for c in strong_counts if c >= 2) / float(n_sampled)) if strong_counts else 0.0
        multi_cat = bool(max_cats >= 2 and multi_ratio >= 0.3)

        result = dict(base)
        result["duration_sec"] = _r3(dur)
        result["frames"] = {"sampled": len(per_frame), "with_cat": len(with_cat)}
        result["scene"] = {
            "multi_cat": multi_cat,
            "max_cats": int(max_cats),
            "multi_frame_ratio": _r3(multi_ratio),
        }

        if not with_cat:
            result["has_cat"] = False
            result["cat"] = None
            result["motion"] = None
            result["note"] = "这段画面里没看到猫 —— 所以我只听了声音。"
            result["tip"] = "想让我看画面，就让猫占画面大一点、光线亮一点。"
            return _emit(result, as_json)

        # --- 猫的几何统计 ---
        ratios = [((b[2] - b[0]) * (b[3] - b[1])) / float(a[2][0] * a[2][1])
                  for a in with_cat for b in [a[1]]]
        confs = [a[0] for a in with_cat]
        big = max(with_cat, key=lambda a: (a[1][2] - a[1][0]) * (a[1][3] - a[1][1]))
        med_ratio = float(np.median(ratios))
        appear = len(with_cat) / float(max(len(per_frame), 1))

        if appear >= 0.9:
            stability = "整段都在画面里"
        elif appear >= 0.5:
            stability = "大部分时间在画面里"
        else:
            stability = "只是偶尔进画面"

        result["has_cat"] = True
        result["cat"] = {
            "appear_ratio": _r3(appear),
            "size_ratio": _r3(med_ratio),
            "size_label": size_label(med_ratio),
            "position": position_label(big[1], big[2][0], big[2][1]),
            "stability": stability,
            "best_conf": _r3(max(confs)),
        }
        # 证据缩略图：把检到猫的那一帧画上框返回，用户可自行核对（比任何文字都诚实）
        try:
            thumb = _thumb(imgs[big[3]], big[1])
            if thumb:
                result["cat"]["thumb"] = thumb
        except Exception:
            pass

        # --- 运动：区分"镜头在动"与"猫在动" ---
        motion = None
        pairs = []
        for i in range(1, len(per_frame)):
            a, b = per_frame[i - 1], per_frame[i]
            if a[1] is None or b[1] is None:
                continue
            g1 = cv2.cvtColor(imgs[a[3]], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(imgs[b[3]], cv2.COLOR_BGR2GRAY)
            if g1.shape != g2.shape:
                continue
            gm = flow_magnitude(g1, g2)
            inter = (max(a[1][0], b[1][0]), max(a[1][1], b[1][1]),
                     min(a[1][2], b[1][2]), min(a[1][3], b[1][3]))
            cm = flow_magnitude(g1, g2, inter) if (inter[2] - inter[0] > 4 and inter[3] - inter[1] > 4) else None
            if gm is not None and cm is not None:
                pairs.append((gm, cm))
        if pairs:
            gm = float(np.mean([p[0] for p in pairs])) / max(step, 0.04)
            cm = float(np.mean([p[1] for p in pairs])) / max(step, 0.04)
            excess = cm - gm
            camera = gm > 12.0
            if camera:
                level = "镜头在动，看不清它自己在不在动"
            elif excess > 8.0:
                level = "它在动"
            elif excess > 2.5:
                level = "它有点小动作"
            else:
                level = "它基本安静地待着"
            motion = {"level": level, "cat_motion_ps": _r3(cm),
                      "frame_motion_ps": _r3(gm), "camera_moving": bool(camera)}
        result["motion"] = motion

        # 诚实说明：只描述画面事实，不解释含义
        result["note"] = ("以上只是画面里能量出来的事实（有没有猫、在哪、多大、动没动）——"
                          "我看不出它在想什么，情绪是听声音得出来的。")
        return _emit(result, as_json)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _emit(obj, as_json):
    if as_json:
        print(json.dumps(obj, ensure_ascii=False))
    else:
        print("=" * 56)
        print("🎬 %s  media=%s" % (obj.get("note", ""), obj.get("media")))
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        print("=" * 56)
    return obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-n", "--frames", type=int, default=None)
    a = ap.parse_args()
    try:
        analyze(a.media, force_n=a.frames, as_json=a.json)
    except Exception as e:      # 任何异常都不该让调用方崩掉
        obj = {"ok": False, "media": "unknown", "has_cat": None,
               "error": "%s: %s" % (type(e).__name__, e),
               "note": "画面没读出来，我只听了声音。"}
        print(json.dumps(obj, ensure_ascii=False))
