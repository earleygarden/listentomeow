# -*- coding: utf-8 -*-
"""
生成「非猫叫」负样本（本机合成，无任何版权问题）
用于训练闸门模型 cat_gate.joblib 的 other 类。

类型：
  noise_white / noise_pink / noise_brown  宽带噪声
  hum_50 / hum_100                        低频电器嗡鸣（50Hz 及谐波）
  tone_*                                  纯音
  sweep_*                                 扫频
  impulse_train                           脉冲串（敲击/咔哒）
  silence_*                               静音（会被 trim 成空）
  chord_*                                 简单正弦和弦（模拟"有人放音乐/视频"）

注意：这些都是「合成音频」，不是真实的音乐或人声，
      用途只是让闸门学会拒绝明显不是猫叫的输入。
      真实的人声负样本请用自己的录音补（见 README）。
"""
import os
import sys
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "synth", "negatives")
SR = 16000


def _save(name, y, sr=SR):
    os.makedirs(OUT, exist_ok=True)
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y / peak * 0.9
    p = os.path.join(OUT, name + ".wav")
    sf.write(p, y, sr, subtype="PCM_16")
    return p, len(y) / sr


def white(dur, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, int(dur * SR))


def _spectral_noise(dur, seed, exponent):
    """用频域整形生成 pink(1/f) / brown(1/f^2) 噪声。"""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    spec = rng.normal(0, 1, n // 2 + 1) + 1j * rng.normal(0, 1, n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spec = spec / (freqs ** exponent)
    return np.fft.irfft(spec, n)


def pink(dur, seed):
    return _spectral_noise(dur, seed, 0.5)


def brown(dur, seed):
    return _spectral_noise(dur, seed, 1.0)


def hum(dur, base):
    t = np.arange(int(dur * SR)) / SR
    y = sum((1.0 / k) * np.sin(2 * np.pi * base * k * t) for k in (1, 2, 3))
    return y * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))  # 轻微起伏


def tone(dur, f):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1.0, np.minimum(t * 20, (dur - t) * 20))
    return np.sin(2 * np.pi * f * t) * np.clip(env, 0, 1)


def sweep(dur, f0, f1):
    t = np.arange(int(dur * SR)) / SR
    k = (f1 / f0) ** (1.0 / dur)
    phase = 2 * np.pi * f0 * (k ** t - 1) / np.log(k)
    return np.sin(phase)


def impulse_train(dur, seed, rate=6.0):
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * SR))
    for _ in range(int(dur * rate)):
        i = rng.integers(0, len(y) - 400)
        y[i:i + 40] += rng.normal(0, 1, 40)
    return y


def chord(dur, seed, base=220.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    n_ch = max(1, int(dur / 1.0))
    for c in range(n_ch):
        f = base * float(2 ** rng.integers(0, 3)) * float(rng.choice([1.0, 1.25, 1.5]))
        seg = (t >= c * 1.0) & (t < (c + 1) * 1.0)
        if not seg.any():
            continue
        tt = t[seg]
        for h in (1, 2, 3):
            out[seg] += np.sin(2 * np.pi * f * h * tt) / h
    return out


def main():
    items = []
    for i, d in enumerate([2.0, 3.5], start=1):
        items.append((f"noise_white_{i:02d}", white(d, 100 + i), "noise_white"))
        items.append((f"noise_pink_{i:02d}", pink(d, 200 + i), "noise_pink"))
        items.append((f"noise_brown_{i:02d}", brown(d, 300 + i), "noise_brown"))
    for b in (50, 100):
        items.append((f"hum_{b}hz", hum(3.0, b), "hum"))
    for f in (440, 1000, 3000):
        for d in (0.6, 1.8):
            items.append((f"tone_{f}hz_{d}s", tone(d, f), "tone"))
    items.append(("sweep_200_4000", sweep(2.0, 200, 4000), "sweep"))
    items.append(("sweep_4000_200", sweep(2.0, 4000, 200), "sweep"))
    items.append(("impulse_train_01", impulse_train(3.0, 11), "impulse"))
    items.append(("impulse_train_02", impulse_train(2.0, 12, rate=12.0), "impulse"))
    for d in (1.0, 3.0):
        items.append((f"silence_{d}s", np.zeros(int(d * SR)), "silence"))
    for i in (1, 2):
        items.append((f"chord_soft_{i:02d}", chord(4.0, 400 + i), "chord"))

    print("生成合成负样本：")
    total = 0.0
    for name, y, _grp in items:
        p, dur = _save(name, y)
        total += dur
        print(f"  {name+'.wav':<26} {dur:5.2f}s")
    print(f"\n共 {len(items)} 个文件 / {total:.1f} 秒 → {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
