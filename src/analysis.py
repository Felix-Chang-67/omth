"""Vocal melody extraction (pYIN) used for the instrumental reprise lead line."""
from __future__ import annotations

import librosa
import numpy as np

from timeline import T


def vocal_notes(vocal_stereo, sr, min_dur=0.07):
    y = librosa.resample(vocal_stereo.mean(axis=1), orig_sr=sr, target_sr=22050)
    hop = 256
    f0, voiced, _ = librosa.pyin(y, fmin=110, fmax=1050, sr=22050, hop_length=hop,
                                 frame_length=1024, max_transition_rate=60)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0][:len(f0)]
    fr = 22050 / hop
    m = np.where(voiced & (np.nan_to_num(f0) > 0) & (rms > 0.008),
                 69 + 12 * np.log2(np.maximum(np.nan_to_num(f0), 1) / 440), np.nan)
    m = np.where(m < 53.5, m + 12, m)          # fold octave errors into the singer's range
    notes, i, n = [], 0, len(m)
    while i < n:
        if np.isnan(m[i]):
            i += 1
            continue
        j, seg = i, [m[i]]
        while j + 1 < n and not np.isnan(m[j + 1]) and abs(m[j + 1] - np.median(seg[-8:])) < 0.6:
            j += 1
            seg.append(m[j])
        dur = (j - i + 1) / fr
        if dur >= min_dur:
            core = np.array(seg[len(seg) // 5: max(len(seg) // 5 + 1, len(seg) * 4 // 5)])
            notes.append(dict(t=i / fr, dur=dur, beats=dur / T, pitch=float(np.median(core))))
        i = j + 1
    return notes
