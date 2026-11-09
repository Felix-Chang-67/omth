"""Small custom synthesizers and effects used on top of the SoundFont instruments."""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

SR = 44100


def _env_exp(n, tau_s):
    return np.exp(-np.arange(n) / (tau_s * SR))


# ---------------------------------------------------------------------------
# drums
# ---------------------------------------------------------------------------
def sub_kick(vel=1.0, f0=150.0, f1=48.0, sweep=0.045, decay=0.28):
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / sweep)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * _env_exp(n, decay)
    click = np.random.default_rng(1).standard_normal(int(0.004 * SR)) * 0.25
    click = ss.lfilter(*ss.butter(2, 3000 / (SR / 2), 'high'), click)
    y[:len(click)] += click * np.linspace(1, 0, len(click))
    fade = int(0.01 * SR)
    y[-fade:] *= np.linspace(1, 0, fade)
    return (vel * y).astype(np.float32)


def place(buf, start_s, sig, gain=1.0):
    i = int(round(start_s * SR))
    if i >= len(buf) or i + len(sig) <= 0:
        return
    j = min(len(buf), i + len(sig))
    if buf.ndim == 2 and sig.ndim == 1:
        buf[i:j] += gain * sig[:j - i, None]
    else:
        buf[i:j] += gain * sig[:j - i]


# ---------------------------------------------------------------------------
# monophonic lead synth (portamento, delayed vibrato, filter envelope)
# ---------------------------------------------------------------------------
def _polyblep_saw(phase, dphase):
    t = phase % 1.0
    y = 2.0 * t - 1.0
    dt = np.maximum(dphase, 1e-9)
    m1 = t < dt
    x = t[m1] / dt[m1]
    y[m1] -= x + x - x * x - 1.0
    m2 = t > 1.0 - dt
    x = (t[m2] - 1.0) / dt[m2]
    y[m2] -= x * x + x + x + 1.0
    return y


def lead_synth(notes, t_start, t_end, glide=0.045, vib_rate=5.6, vib_cents=14, vib_delay=0.22):
    """notes: list of (start_s, dur_s, midi, velocity) in absolute seconds.
    Returns audio covering [t_start, t_end)."""
    n = int((t_end - t_start) * SR)
    notes = [(t - t_start, d, p, v) for t, d, p, v in notes]
    f = np.zeros(n)
    amp = np.zeros(n)
    cut = np.zeros(n)
    notes = sorted(notes)
    prev_f = None
    for k, (t0, d, p, v) in enumerate(notes):
        i0, i1 = int(t0 * SR), min(n, int((t0 + d) * SR))
        if i1 <= i0:
            continue
        target = 440.0 * 2 ** ((p - 69) / 12)
        seg = np.full(i1 - i0, target)
        if prev_f is not None and k > 0 and t0 - (notes[k - 1][0] + notes[k - 1][1]) < 0.06:
            g = min(len(seg), int(glide * SR))
            seg[:g] = prev_f * (target / prev_f) ** (np.arange(g) / max(g, 1))
        tt = np.arange(i1 - i0) / SR
        depth = np.clip((tt - vib_delay) / 0.25, 0, 1) * vib_cents
        seg = seg * 2 ** (depth * np.sin(2 * np.pi * vib_rate * tt) / 1200)
        f[i0:i1] = seg
        a = np.ones(i1 - i0)
        att = min(len(a), int(0.012 * SR))
        a[:att] = np.linspace(0, 1, att)
        a *= 0.82 + 0.18 * np.exp(-tt / 0.18)
        rel = min(len(a), int(0.05 * SR))
        a[-rel:] *= np.linspace(1, 0, rel)
        amp[i0:i1] = a * (v / 127)
        cut[i0:i1] = 1800 + 3800 * np.exp(-tt / 0.25)
        prev_f = target
    # hold frequency during gaps to avoid phase jumps
    idx = np.where(f > 0, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    f = f[idx]
    f[f == 0] = 220.0
    out = np.zeros(n)
    for det in (-0.07, 0.07):
        fd = f * 2 ** (det / 12)
        dph = fd / SR
        ph = np.cumsum(dph)
        saw = _polyblep_saw(ph, dph)
        sq = np.sign(np.sin(2 * np.pi * (ph + 0.5 * det)))
        out += 0.7 * saw + 0.18 * sq
    out *= amp
    # time-varying one-pole low-pass approximated in blocks (filter envelope)
    y = np.zeros(n)
    z = 0.0
    blk = 64
    for s in range(0, n, blk):
        fc = cut[s] if cut[s] > 0 else 1800
        a = np.exp(-2 * np.pi * fc / SR)
        seg = out[s:s + blk]
        o, zf = ss.lfilter([1 - a], [1, -a], seg, zi=[z])
        y[s:s + blk] = o
        z = zf[0]
    y = ss.lfilter(*ss.butter(2, 120 / (SR / 2), 'high'), y)
    return (0.5 * y).astype(np.float32)


# ---------------------------------------------------------------------------
# transition FX
# ---------------------------------------------------------------------------
def reverse_cymbal(dur=1.6, seed=3):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = rng.standard_normal((n, 2))
    noise = ss.lfilter(*ss.butter(2, 4500 / (SR / 2), 'high'), noise, axis=0)
    env = (np.arange(n) / n) ** 3.2
    y = noise * env[:, None]
    y[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))[:, None]
    return (0.35 * y / np.abs(y).max()).astype(np.float32)


def riser(dur=4.0, seed=5):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = rng.standard_normal((n, 2))
    out = np.zeros_like(noise)
    blk = 2048
    for s in range(0, n, blk):
        x = s / n
        fc = 300 * (40 ** x)
        b, a = ss.butter(2, [min(fc, 15000) / (SR / 2) * 0.7, min(fc * 1.6, 19000) / (SR / 2)], 'band')
        out[s:s + blk] = ss.lfilter(b, a, noise[s:s + blk], axis=0)
    env = (np.arange(n) / n) ** 2.0
    out *= env[:, None]
    out[-int(0.003 * SR):] *= 0
    return (0.3 * out / (np.abs(out).max() + 1e-9)).astype(np.float32)


def sub_drop(dur=1.4):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 32 + 60 * np.exp(-t / 0.18)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45)
    y[:int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    return (0.8 * y).astype(np.float32)


def vinyl(dur, seed=7):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    hiss = ss.lfilter(*ss.butter(2, [900 / (SR / 2), 7000 / (SR / 2)], 'band'), rng.standard_normal(n)) * 0.004
    crack = np.zeros(n)
    idx = rng.integers(0, n, int(dur * 9))
    crack[idx] = np.clip(rng.standard_normal(len(idx)), -2.5, 2.5) * rng.uniform(0.004, 0.02, len(idx))
    crack = ss.lfilter(*ss.butter(2, 2500 / (SR / 2), 'high'), crack)
    crack = ss.lfilter(*ss.butter(1, 9000 / (SR / 2), 'low'), crack)       # soften the clicks
    rumble = ss.lfilter(*ss.butter(2, 60 / (SR / 2), 'low'), rng.standard_normal(n)) * 0.003
    y = hiss + crack + rumble
    return np.stack([y, np.roll(y, 37)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# synthetic stereo reverb impulse responses
# ---------------------------------------------------------------------------
def make_ir(t60_bands=(2.4, 2.1, 1.7, 1.1), predelay=0.02, length=None, seed=11, early=True):
    """Exponentially decaying decorrelated noise, with frequency-dependent decay."""
    rng = np.random.default_rng(seed)
    length = length or max(t60_bands) * 1.2
    n = int(length * SR)
    edges = [20, 500, 2000, 6000, 20000]
    ir = np.zeros((n, 2))
    t = np.arange(n) / SR
    for (lo, hi), t60 in zip(zip(edges[:-1], edges[1:]), t60_bands):
        noise = rng.standard_normal((n, 2))
        b, a = ss.butter(2, [lo / (SR / 2), min(hi, 19999) / (SR / 2)], 'band')
        band = ss.lfilter(b, a, noise, axis=0)
        ir += band * (10 ** (-3 * t / t60))[:, None]
    att = int(0.006 * SR)
    ir[:att] *= np.linspace(0, 1, att)[:, None]
    if early:
        for k in range(8):
            d = int((0.007 + 0.011 * k + rng.uniform(0, 0.004)) * SR)
            g = 0.5 * (0.8 ** k)
            ir[d, 0] += g * rng.choice([-1, 1])
            ir[d + int(rng.uniform(0, 0.003) * SR), 1] += g * rng.choice([-1, 1])
    pd = int(predelay * SR)
    ir = np.concatenate([np.zeros((pd, 2)), ir])
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    return ir.astype(np.float32)


def convolve(x, ir):
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    out = np.stack([ss.oaconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], axis=1)
    return out.astype(np.float32)
