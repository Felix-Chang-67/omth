"""Draws the arrangement map (sections, chords, instrument activity, loudness).

usage: python3 plot_arrangement.py <out.png> [<rendered master.wav>]
"""
from __future__ import annotations

import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from chords import SECTIONS, chord_events  # noqa: E402
from parts import all_parts  # noqa: E402
from timeline import BAR, SR, out_time  # noqa: E402

COLORS = {
    'pre': '#2b2d42', 'intro_v': '#3d5a80', 'intro_b': '#ee6c4d', 'verse1': '#98c1d9', 'pre1': '#e0aaff',
    'chorus': '#ff006e', 'vamp': '#fb8500', 'verse2': '#8ecae6', 'pre2': '#c77dff', 'chorus2': '#ff006e',
    'vamp2': '#fb8500', 'bridge': '#264653', 'build': '#9d4edd', 'final': '#d00000', 'ending': '#e85d04',
    'tag': '#3d5a80', 'reprise': '#ffb703', 'end': '#2b2d42',
}
ORDER = ['vocal (original)', 'lead', 'hook', 'brass', 'synth_brass', 'brass_swell', 'strings', 'violins', 'choir',
         'pad', 'vibes', 'glock', 'rhodes', 'guitar', 'bass', 'kick', 'snare', 'clap', 'hats', 'cymbals', 'toms', 'perc']


def main(out_png, master_wav=None):
    parts = {p.name: p for p in all_parts()}
    fig = plt.figure(figsize=(26, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.7, 7.0, 1.6], hspace=0.08)
    ax_s, ax_c, ax_a, ax_l = [fig.add_subplot(g) for g in gs]
    x0, x1 = -4, 175
    for s in SECTIONS:
        ax_s.add_patch(plt.Rectangle((s['start'], 0), s['end'] - s['start'], 1, color=COLORS[s['kind']], alpha=0.9))
        ax_s.text((s['start'] + s['end']) / 2, 0.5, s['name'].replace('_', '\n'), ha='center', va='center',
                  fontsize=8.5, color='white', fontweight='bold')
    ax_s.set_xlim(x0, x1)
    ax_s.set_ylim(0, 1)
    ax_s.axis('off')
    ax_s.set_title('One Meets Her - Midnight City Pop rearrangement (x axis: original bars @ 191 BPM = half bars of the new 95.5 BPM groove)',
                   fontsize=13)
    last = None
    for b, e, c in chord_events():
        if c.is_nc or b < x0:
            continue
        ax_c.axvline(b, color='#999999', lw=0.4)
        if c.symbol != last and (e - b) >= 0.9:
            ax_c.text(b + 0.05, 0.5, c.symbol, fontsize=6.3, rotation=90, va='center')
        last = c.symbol
    ax_c.set_xlim(x0, x1)
    ax_c.set_ylim(0, 1)
    ax_c.axis('off')
    rows = []
    for name in ORDER:
        act = np.zeros(int((x1 - x0) * 4))
        if name == 'vocal (original)':
            for s in SECTIONS:
                if s['kind'] not in ('pre', 'reprise', 'end') and not s['name'].startswith('interlude'):
                    i0, i1 = int((s['start'] - x0) * 4), int((s['end'] - x0) * 4)
                    act[i0:i1] = 0.8
            act[int((8.3 - x0) * 4):int((16.3 - x0) * 4)] = 0
        elif name == 'lead':
            act[int((154.5 - x0) * 4):int((173 - x0) * 4)] = 0.9
        else:
            p = parts.get(name)
            if p is None:
                continue
            for n in p.notes:
                b = (n.t - out_time(0)) / BAR
                i0, i1 = int((b - x0) * 4), int((b + n.d / BAR - x0) * 4) + 1
                act[max(0, i0):max(0, i1)] = np.maximum(act[max(0, i0):max(0, i1)], n.v / 127)
        rows.append((name, act))
    img = np.array([r[1] for r in rows])
    ax_a.imshow(img, aspect='auto', cmap='magma', extent=[x0, x1, len(rows), 0], vmin=0, vmax=1, interpolation='nearest')
    ax_a.set_yticks(np.arange(len(rows)) + 0.5)
    ax_a.set_yticklabels([r[0] for r in rows], fontsize=10)
    for s in SECTIONS:
        ax_a.axvline(s['start'], color='cyan', lw=0.8, alpha=0.7)
    ax_a.set_xlim(x0, x1)
    ax_a.tick_params(labelbottom=False)
    if master_wav:
        import soundfile as sf
        y, _ = sf.read(master_wav, dtype='float32')
        y = y.mean(axis=1)
        hop = int(0.4 * SR)
        n = len(y) // hop
        rms = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
        t = (np.arange(n) * hop / SR - out_time(0)) / BAR
        ax_l.fill_between(t, -40, 20 * np.log10(rms + 1e-9), color='#ff006e', alpha=0.6)
        ax_l.set_ylim(-40, -5)
        ax_l.set_ylabel('RMS dBFS')
    ax_l.set_xlim(x0, x1)
    ax_l.set_xlabel('original bar')
    for s in SECTIONS:
        ax_l.axvline(s['start'], color='#888888', lw=0.6)
    fig.savefig(out_png, dpi=70, bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
