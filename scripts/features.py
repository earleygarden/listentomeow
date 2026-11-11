# -*- coding: utf-8 -*-
"""
猫叫声学特征提取模块 (v2.1)
- 统一采样率 16kHz、单声道；去首尾静音；最短 0.5s、最长 30s
- 音频解码：优先 soundfile(wav/mp3/flac/ogg)，失败则用 imageio-ffmpeg 兜底(m4a/aac/mp4/webm/amr...)
- 特征组：
    * MFCC(20) + 一阶差分 + 二阶差分  → 各 4 项统计
    * 谱质心 / 谱带宽 / 谱滚降 / 谱平坦度 / 过零率 / RMS → 各 4 项统计
    * 基频 f0 (yin) → 4 项统计
- 输出 268 维特征向量，对音频长度不敏感（统计池化）
"""
import os
import numpy as np
import librosa

SR = 16000
N_MFCC = 20
N_FFT = 2048
HOP = 512
FMIN = 50
FMAX = 7500
MIN_SEC = 0.5      # 最短有效时长
MAX_SEC = 30.0     # 最长处理时长（超出部分截断，避免超长录音拖慢/占内存）


def _load_via_ffmpeg(path, sr=SR):
    """用 imageio-ffmpeg 自带的 ffmpeg 解码为 wav 再读，兜底所有常见格式。
    -vn 明确丢弃视频流（视频容器里我们只要音轨，也避免无谓的解码开销）。"""
    import subprocess
    import tempfile
    import imageio_ffmpeg
    import soundfile as sf

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        r = subprocess.run(
            [ffmpeg, "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", tmp],
            capture_output=True, timeout=90,
        )
        if r.returncode != 0 or not os.path.exists(tmp):
            detail = r.stderr.decode("utf-8", "ignore")[-300:]
            raise RuntimeError(f"无法解码该音频（ffmpeg 失败）：{detail}")
        y, _ = sf.read(tmp, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# 视频容器扩展名：这些格式 librosa 的 soundfile 路径一律不支持，
# 直接交给 ffmpeg 抽音轨，比"先失败再兜底"更快，也避免无音轨视频走到半路才炸。
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv",
             ".flv", ".mpg", ".mpeg", ".3gp", ".ts", ".mts"}


def load_audio(path, sr=SR):
    """读取音频为单声道、统一采样率。
    视频容器直接走内置 ffmpeg 抽音轨；音频优先 soundfile（wav/mp3/flac/ogg），
    失败则用 ffmpeg 转码，以兼容 m4a / aac / webm / amr 等手机录音格式。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXT:
        return _load_via_ffmpeg(path, sr)
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y
    except Exception:
        return _load_via_ffmpeg(path, sr)


def trim_silence(y, top_db=30):
    if len(y) == 0:
        return y
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > 0 else y


def _stats(m):
    """沿时间轴的 4 项统计（mean/std/min/max）。"""
    m = np.atleast_2d(m)
    return np.concatenate([m.mean(1), m.std(1), m.min(1), m.max(1)])


def extract_features_from_signal(y, sr=SR):
    """对已加载波形提取 268 维特征。"""
    y = trim_silence(y)
    max_len = int(MAX_SEC * sr)
    if len(y) > max_len:
        y = y[:max_len]
    min_len = int(MIN_SEC * sr)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                hop_length=HOP, fmin=FMIN, fmax=min(FMAX, sr // 2 - 1))
    d1 = librosa.feature.delta(mfcc, width=3)
    d2 = librosa.feature.delta(mfcc, order=2, width=3)

    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    flat = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)

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


def extract_features(path, sr=SR):
    """从音频文件路径提取特征。"""
    return extract_features_from_signal(load_audio(path, sr=sr), sr=sr)


FEATURE_DIM = N_MFCC * 3 * 4 + 6 * 4 + 4  # 240 + 24 + 4 = 268

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1:
        v = extract_features(sys.argv[1])
        print("特征维度:", v.shape, "| 前5维:", np.round(v[:5], 3))
    else:
        print("用法: python features.py <音频文件>")
