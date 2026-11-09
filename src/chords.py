"""Chord vocabulary, the reharmonised chord chart and the section map.

Positions are in original bars (see timeline.py).  A chord lasts until the next
entry.  'NC' means no chord (band stop).
"""
from __future__ import annotations

import re

PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6,
      'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']

# quality: chord tones, available tensions, avoid notes (intervals above root),
#          keys voicing (4 intervals, rootless where idiomatic), guitar upper-structure triad
QUALITIES = {
    '':        dict(tones=[0, 4, 7],          tens=[2],        avoid=[5],    keys=[0, 4, 7, 12],  gtr=[0, 4, 7]),
    'm':       dict(tones=[0, 3, 7],          tens=[2, 5],     avoid=[8],    keys=[0, 3, 7, 12],  gtr=[0, 3, 7]),
    'add9':    dict(tones=[0, 4, 7, 2],       tens=[9],        avoid=[5],    keys=[4, 7, 0, 2],   gtr=[4, 7, 2]),
    '6':       dict(tones=[0, 4, 7, 9],       tens=[2],        avoid=[5],    keys=[4, 7, 9, 0],   gtr=[4, 7, 9]),
    '69':      dict(tones=[0, 4, 7, 9, 2],    tens=[],         avoid=[5],    keys=[4, 9, 2, 7],   gtr=[4, 9, 2]),
    'maj7':    dict(tones=[0, 4, 7, 11],      tens=[2, 6, 9],  avoid=[5],    keys=[4, 7, 11, 2],  gtr=[4, 7, 11]),
    'maj9':    dict(tones=[0, 4, 7, 11, 2],   tens=[6, 9],     avoid=[5],    keys=[4, 7, 11, 2],  gtr=[4, 7, 11]),
    'maj7#11': dict(tones=[0, 4, 7, 11, 6],   tens=[2, 9],     avoid=[5],    keys=[4, 6, 11, 2],  gtr=[2, 6, 9]),
    'm7':      dict(tones=[0, 3, 7, 10],      tens=[2, 5],     avoid=[8],    keys=[3, 7, 10, 0],  gtr=[3, 7, 10]),
    'm9':      dict(tones=[0, 3, 7, 10, 2],   tens=[5],        avoid=[8],    keys=[3, 7, 10, 2],  gtr=[7, 10, 2]),
    'm11':     dict(tones=[0, 3, 7, 10, 2, 5], tens=[],        avoid=[8],    keys=[3, 10, 2, 5],  gtr=[10, 2, 5]),
    'm7b5':    dict(tones=[0, 3, 6, 10],      tens=[5, 8],     avoid=[1],    keys=[3, 6, 10, 0],  gtr=[3, 6, 10]),
    '7':       dict(tones=[0, 4, 7, 10],      tens=[2, 9, 1, 3, 8], avoid=[5], keys=[4, 10, 0, 7], gtr=[4, 10, 7]),
    '9':       dict(tones=[0, 4, 7, 10, 2],   tens=[9],        avoid=[5],    keys=[4, 10, 2, 7],  gtr=[4, 10, 2]),
    '13':      dict(tones=[0, 4, 7, 10, 2, 9], tens=[],        avoid=[5],    keys=[4, 10, 2, 9],  gtr=[4, 10, 9]),
    '7sus4':   dict(tones=[0, 5, 7, 10],      tens=[2, 9],     avoid=[4],    keys=[5, 10, 0, 7],  gtr=[5, 10, 0]),
    '9sus4':   dict(tones=[0, 5, 7, 10, 2],   tens=[9],        avoid=[4],    keys=[5, 10, 2, 7],  gtr=[10, 2, 5]),
    '13sus4':  dict(tones=[0, 5, 7, 10, 2, 9], tens=[],        avoid=[4],    keys=[5, 10, 2, 9],  gtr=[10, 2, 5]),
    '7#9':     dict(tones=[0, 4, 7, 10, 3],   tens=[1, 8],     avoid=[5],    keys=[4, 10, 3, 7],  gtr=[4, 10, 3]),
    '7b9':     dict(tones=[0, 4, 7, 10, 1],   tens=[3, 8],     avoid=[5],    keys=[4, 10, 1, 7],  gtr=[4, 10, 1]),
    '7b13':    dict(tones=[0, 4, 7, 10, 8],   tens=[1, 3],     avoid=[5],    keys=[4, 10, 8, 0],  gtr=[4, 10, 8]),
    '7#9b13':  dict(tones=[0, 4, 7, 10, 3, 8], tens=[1],       avoid=[5],    keys=[4, 10, 3, 8],  gtr=[4, 10, 3]),
}

_SYM = re.compile(r'^([A-G](?:#|b)?)([^/]*)(?:/([A-G](?:#|b)?))?$')


class Chord:
    def __init__(self, symbol: str):
        self.symbol = symbol
        if symbol == 'NC':
            self.root = None
            return
        m = _SYM.match(symbol)
        if not m:
            raise ValueError(symbol)
        self.root = PC[m.group(1)]
        q = m.group(2)
        if q not in QUALITIES:
            raise ValueError(f'unknown quality {q!r} in {symbol}')
        self.quality = q
        self.q = QUALITIES[q]
        self.bass = PC[m.group(3)] if m.group(3) else self.root

    @property
    def is_nc(self):
        return self.root is None

    def pcs(self, key):
        return [(self.root + i) % 12 for i in self.q[key]]

    def tone_pcs(self):
        return sorted(set(self.pcs('tones')))

    def __repr__(self):
        return self.symbol


# ---------------------------------------------------------------------------
# Reharmonised chord chart (original bar -> chord)
# ---------------------------------------------------------------------------
CHORUS = [  # 16 bars, relative to chorus start (王道進行 + altered dominants)
    (0, 'Fmaj9'), (1, 'G13sus4'), (2, 'Cadd9/E'), (3, 'Am9'),
    (4, 'Fmaj9'), (5, 'G9sus4'), (6, 'E7b13'), (7, 'Am9'),
    (8, 'Fmaj7'), (9, 'E7b13'), (10, 'G9sus4'), (11, 'Am9'),
    (12, 'Fmaj9'), (13, 'G13sus4'), (14, 'E7#9b13'), (15, 'Am9'),
]
PRECHORUS = [  # 8 bars: minor line cliche D-C-B-Bb, then D E F E
    (0, 'Dm9'), (1, 'Dm9/C'), (2, 'Bm7b5'), (3, 'Bbmaj7#11'),
    (4, 'Dm9'), (5, 'E7sus4'), (5.5, 'E7'), (6, 'Fmaj7'), (7, 'E7#9'),
]
VERSE1 = [  # 16 bars (original: Am | G every 2 bars)
    (0, 'Am9'), (2, 'Fmaj7'), (3, 'G6'), (4, 'Am9'), (5, 'Am9/G'), (6, 'Fmaj7'), (7, 'E7#9'),
    (8, 'Am9'), (10, 'Fmaj7'), (11, 'G6'), (12, 'Am9'), (13, 'Am9/G'), (14, 'Fmaj7'),
    (15, 'E7sus4'), (15.5, 'E7'),
]
VERSE2 = [  # 16 bars (original: Fmaj7#11 / E7sus4 vamp, then F | C/E)
    (0, 'Fmaj9'), (1, 'E7sus4'), (2, 'E7'), (3, 'Am9'),
    (4, 'Fmaj9'), (5, 'E7sus4'), (6, 'E7b9'), (7, 'Am9'),
    (8, 'Fmaj9'), (10, 'Am9/E'), (12, 'Fmaj9'), (14, 'Eadd9'), (15, 'E7'),
]
VAMP = [(0, 'Am9'), (1, 'D9'), (2, 'Am9'), (3, 'D9')]


def _at(base, rel):
    return [(base + b, c) for b, c in rel]


CHART = (
    [(-4, 'Am9'), (-2, 'E7sus4'), (-1, 'E7b9')]                                  # new Rhodes pre-intro
    + [(0, 'Fmaj9'), (2, 'Dm9'), (4, 'Bbmaj9'), (6, 'Am9'), (7.75, 'E7#9')]   # vocal intro
    + [(8, 'Fmaj9'), (9, 'G13sus4'), (10, 'Cadd9/E'), (11, 'Am9'),
       (12, 'Fmaj9'), (13, 'G13'), (14, 'E7sus4'), (15, 'E7#9')]                  # band intro (+ 2/4 bar)
    + _at(16.5, VERSE1)
    + _at(32.5, PRECHORUS) + [(40.5, 'NC')]
    + _at(41, CHORUS)
    + _at(57, VAMP)
    + _at(61, VERSE2)
    + _at(77, PRECHORUS[:-1]) + [(84, 'G7'), (84.25, 'NC')]
    + _at(85, CHORUS)
    + _at(101, VAMP)
    + [(105, 'Am11'), (107, 'G/A'), (109, 'Fmaj7/A'), (111, 'Dm9/A'),
       (113, 'Am11'), (115, 'G/A'), (117, 'Fmaj7/A'), (119, 'A7sus4'), (120, 'A7b9')]   # bridge
    + [(121, 'Dm9'), (122, 'E7#9'), (123, 'Fmaj9'), (124, 'G69'),
       (125, 'Fmaj7'), (126, 'E7b9'), (127, 'Am11'), (128, 'E7b13')]              # build
    + _at(129, CHORUS[:12])
    + [(141, 'Fmaj9'), (142, 'Dm9'), (143, 'G13sus4'), (144, 'G13')]            # climax (C5 sus -> B)
    + [(145, 'Am11'), (146, 'G13sus4'), (147, 'E7b13'), (148, 'Am11'),
       (149, 'Fmaj7'), (150, 'G6'), (150.5, 'E7')]                                  # ending phrase (bass walk A G# F# E)
    + [(152, 'E7sus4'), (153, 'E7b9')]                                              # vocal tag (keys only)
    + _at(155, CHORUS[:13]) + [(169, 'E7sus4'), (169.5, 'E7#9b13')]                  # instrumental reprise
    + [(171, 'Am9'), (173, 'NC')]                                                   # final chord
)

# ---------------------------------------------------------------------------
# Section map (original bars).  `kind` selects the patterns of every part.
# ---------------------------------------------------------------------------
SECTIONS = [
    dict(name='pre_intro', start=-4, end=0, kind='pre'),
    dict(name='intro_vocal', start=0, end=8, kind='intro_v'),
    dict(name='intro_band', start=8, end=16.5, kind='intro_b'),
    dict(name='verse1', start=16.5, end=32.5, kind='verse1'),
    dict(name='prechorus1', start=32.5, end=41, kind='pre1'),
    dict(name='chorus1', start=41, end=57, kind='chorus'),
    dict(name='interlude1', start=57, end=61, kind='vamp'),
    dict(name='verse2', start=61, end=77, kind='verse2'),
    dict(name='prechorus2', start=77, end=85, kind='pre2'),
    dict(name='chorus2', start=85, end=101, kind='chorus2'),
    dict(name='interlude2', start=101, end=105, kind='vamp2'),
    dict(name='bridge', start=105, end=121, kind='bridge'),
    dict(name='build', start=121, end=129, kind='build'),
    dict(name='final_chorus', start=129, end=145, kind='final'),
    dict(name='ending', start=145, end=151, kind='ending'),
    dict(name='tag', start=151, end=155, kind='tag'),
    dict(name='reprise', start=155, end=171, kind='reprise'),
    dict(name='final_chord', start=171, end=173, kind='end'),
]


def chord_events():
    """List of (start_bar, end_bar, Chord)."""
    ev = []
    for i, (b, s) in enumerate(CHART):
        e = CHART[i + 1][0] if i + 1 < len(CHART) else b + 2
        if e <= b:
            raise ValueError(f'chart not increasing at {b}')
        ev.append((b, e, Chord(s)))
    return ev


def chord_at(bar):
    for b, e, c in chord_events():
        if b <= bar < e:
            return c
    return Chord('NC')


def section_at(bar):
    for s in SECTIONS:
        if s['start'] <= bar < s['end']:
            return s
    return None
