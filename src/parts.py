"""Generates every new instrumental part of the rearrangement as note data.

Times are in seconds on the output timeline (see timeline.py).  Each part is a
list of notes plus optional controller changes; render.py turns them into
audio.  Patterns are written on the 16-step grid of the new 95.5 BPM groove
(one step = one 8th note of the original 191 BPM song).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

from chords import SECTIONS, Chord, chord_events
from timeline import STEP, out_time

RNG = random.Random(20261109)


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass
class Note:
    t: float
    d: float
    p: int
    v: int


@dataclass
class Part:
    name: str
    program: int = 0
    bank: int = 0
    drum: bool = False
    notes: list = field(default_factory=list)
    cc: list = field(default_factory=list)      # (t, controller, value)

    def add(self, t, d, p, v):
        if d <= 0.005:
            return
        self.notes.append(Note(t, d, int(p), int(max(1, min(127, round(v))))))

    def ctrl(self, t, num, val):
        self.cc.append((t, num, int(max(0, min(127, round(val))))))


def jitter(sd):
    return RNG.gauss(0.0, sd)


def vel(v, sd=4.0):
    return v + RNG.gauss(0.0, sd)


# ---------------------------------------------------------------------------
# chord / section helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _events():
    return tuple(chord_events())


def chord_at(bar):
    for b, e, c in _events():
        if b <= bar + 1e-6 < e:
            return c
    return Chord('NC')


def next_change(bar):
    for b, e, c in _events():
        if b <= bar + 1e-6 < e:
            return e
    return bar + 2


def spans(b0, b1):
    """Chord spans clipped to [b0, b1): list of (start, end, chord)."""
    out = []
    for b, e, c in _events():
        s, t = max(b, b0), min(e, b1)
        if t - s > 1e-6:
            out.append((s, t, c))
    return out


def new_bars(sec):
    """Split a section into new (2-original-bar) bars: (bar, n_steps, idx, count)."""
    res = []
    b = sec['start']
    while b < sec['end'] - 1e-6:
        e = min(b + 2, sec['end'])
        res.append([b, int(round((e - b) * 8))])
        b = e
    return [(b, n, i, len(res)) for i, (b, n) in enumerate(res)]


def t_of(bar, step=0.0):
    return out_time(bar) + step * STEP


def kind_of(sec):
    return sec['kind']


# ---------------------------------------------------------------------------
# voicing engine (smooth voice-leading inside a register)
# ---------------------------------------------------------------------------
def voice(pcs, prev, lo, hi, max_span=16, center=None):
    pcs = list(dict.fromkeys(pcs))
    options = [[p for p in range(lo, hi + 1) if p % 12 == pc] for pc in pcs]
    if any(len(o) == 0 for o in options):
        raise ValueError((pcs, lo, hi))
    c0 = center if center is not None else (lo + hi) / 2
    best, bscore = None, 1e18
    for combo in product(*options):
        notes = sorted(combo)
        if len(set(notes)) != len(notes) or notes[-1] - notes[0] > max_span:
            continue
        sc = 0.0
        for a, b in zip(notes, notes[1:]):
            if b - a == 1:
                sc += 4.0
        if notes[0] < 53 and len(notes) > 1 and notes[1] - notes[0] < 4:
            sc += 6.0                                    # muddy low clusters
        if prev:
            pv = sorted(prev)
            if len(pv) == len(notes):
                sc += sum(abs(x - y) for x, y in zip(pv, notes))
            else:
                sc += 2 * abs(sum(pv) / len(pv) - sum(notes) / len(notes))
        sc += 0.35 * abs(sum(notes) / len(notes) - c0)
        if sc < bscore:
            best, bscore = notes, sc
    return best


class Voicer:
    def __init__(self, key='keys', lo=52, hi=76, max_span=16, center=None, extra_top=False):
        self.key, self.lo, self.hi, self.max_span, self.center = key, lo, hi, max_span, center
        self.prev = None
        self.cache = {}
        self.extra_top = extra_top

    def __call__(self, chord: Chord):
        if chord.is_nc:
            return []
        v = voice(chord.pcs(self.key), self.prev, self.lo, self.hi, self.max_span, self.center)
        self.prev = v
        if self.extra_top:
            v = v + [v[-1] + 12]
        return v


def bass_pitch(pc, prev=None, lo=28, hi=40):
    cands = [p for p in range(lo, hi + 1) if p % 12 == pc]
    ref = prev if prev is not None else 34
    return min(cands, key=lambda p: abs(p - ref) + 0.25 * abs(p - 34))


# ===========================================================================
# DRUMS (General MIDI kit, rendered per voice for mixing)
# ===========================================================================
KICK, SNARE, RIM, CLAP, HHC, HHP, HHO = 36, 38, 37, 39, 42, 44, 46
CRASH, CRASH2, RIDE, RIDEBELL, SPLASH = 49, 57, 51, 53, 55
TOM_FL, TOM_L, TOM_M, TOM_H = 43, 45, 47, 50
SHAKER, TAMB = 82, 54

DRUM_VOICE = {KICK: 'kick', SNARE: 'snare', RIM: 'snare', CLAP: 'clap', HHC: 'hats', HHP: 'hats',
              HHO: 'hats', CRASH: 'cymbals', CRASH2: 'cymbals', RIDE: 'cymbals', RIDEBELL: 'cymbals',
              SPLASH: 'cymbals', TOM_FL: 'toms', TOM_L: 'toms', TOM_M: 'toms', TOM_H: 'toms',
              SHAKER: 'perc', TAMB: 'perc'}

GROOVES = {
    'disco': {   # four-on-the-floor city-pop / boogie groove
        KICK: [(0, 112), (4, 102), (8, 110), (12, 102)],
        SNARE: [(4, 100), (12, 106)],
        CLAP: [(4, 74), (12, 80)],
        HHC: [(s, 70 if s % 4 == 0 else 50 if s % 2 == 0 else 40) for s in range(16) if s not in (2, 6, 10, 14)],
        HHO: [(2, 62), (6, 60), (10, 62), (14, 66)],
        SHAKER: [(s, 46 if s % 2 == 0 else 32) for s in range(16)],
    },
    'verse1': {
        KICK: [(0, 100), (7, 72), (10, 90)],
        RIM: [(4, 82), (12, 86)],
        HHC: [(s, 60 if s % 4 == 0 else 46) for s in range(0, 16, 2)] + [(15, 34)],
        SHAKER: [(s, 36 if s % 2 == 0 else 24) for s in range(16)],
    },
    'verse2': {
        KICK: [(0, 104), (6, 78), (10, 94)],
        SNARE: [(4, 90), (12, 96)],
        HHC: [(s, 64 if s % 4 == 0 else 50 if s % 2 == 0 else 38) for s in range(16)],
        SHAKER: [(s, 40 if s % 2 == 0 else 28) for s in range(16)],
    },
    'pre': {
        KICK: [(0, 106), (8, 100), (11, 72)],
        SNARE: [(4, 96), (12, 100)],
        HHC: [(s, 64 if s % 4 == 0 else 50) for s in range(0, 16, 2)],
        SHAKER: [(s, 40 if s % 2 == 0 else 28) for s in range(16)],
    },
    'bridge_a': {
        KICK: [(0, 74)],
        RIM: [(4, 44), (12, 48)],
    },
    'bridge_b': {
        KICK: [(0, 88), (10, 72)],
        RIM: [(4, 64), (12, 68)],
        HHC: [(s, 52 if s % 4 == 0 else 40) for s in range(0, 16, 2)],
        SHAKER: [(s, 38 if s % 2 == 0 else 26) for s in range(16)],
    },
}

FILL_TOMS = [(8, SNARE, 92), (9, SNARE, 70), (10, TOM_H, 96), (11, TOM_H, 84), (12, TOM_M, 98),
             (13, TOM_M, 86), (14, TOM_L, 102), (15, TOM_FL, 108)]
FILL_SNARE = [(12, SNARE, 74), (13, SNARE, 82), (14, SNARE, 92), (15, SNARE, 104)]


def drums():
    parts = {k: Part(k, program=8, drum=True) for k in ['kick', 'snare', 'clap', 'hats', 'cymbals', 'toms', 'perc']}

    def hit(t, note, v, sd=0.003):
        name = DRUM_VOICE[note]
        dur = 0.9 if name == 'cymbals' else 0.25
        parts[name].add(t + jitter(sd), dur, note, vel(v, 3))

    def groove(b, steps, name, skip=(), vscale=1.0, extra=None):
        pat = GROOVES[name]
        for note, hits in pat.items():
            for s, v in hits:
                if s < steps and s not in skip:
                    hit(t_of(b, s), note, v * vscale, 0.004 if note in (HHC, HHO, SHAKER) else 0.0025)
        if extra:
            for s, note, v in extra:
                if s < steps:
                    hit(t_of(b, s), note, v)

    def fill(b, kind='toms'):
        f = FILL_TOMS if kind == 'toms' else FILL_SNARE
        for s, note, v in f:
            hit(t_of(b, s), note, v)
        return {s for s, _, _ in f}

    for sec in SECTIONS:
        k = kind_of(sec)
        nbs = new_bars(sec)
        ends_with_insert = nbs[-1][1] < 16
        for b, steps, idx, count in nbs:
            last = (idx == count - 1) or (ends_with_insert and idx == count - 2)
            if k == 'tag' and idx == 0:           # band stop hit before the vocal tag
                hit(t_of(b, 0), KICK, 112)
                hit(t_of(b, 0), CRASH, 108)
                hit(t_of(b, 0), SNARE, 90)
            if k in ('pre', 'tag'):
                if k == 'tag' and last:          # pickup into the reprise
                    for s, note, v in [(8, SNARE, 70), (10, SNARE, 80), (12, TOM_H, 90), (13, TOM_M, 92),
                                       (14, TOM_L, 98), (15, TOM_FL, 106)]:
                        hit(t_of(b, s), note, v)
                continue
            if k == 'intro_v':
                if idx >= 2:                      # shaker creeps in before the band
                    for s in range(16):
                        hit(t_of(b, s), SHAKER, (24 + 14 * (idx - 2) + s) * (1.0 if s % 2 == 0 else 0.7))
                continue
            if k == 'end':
                if idx == 0:
                    hit(t_of(b, 0), KICK, 112)
                    hit(t_of(b, 0), CRASH, 110)
                    hit(t_of(b, 0), CRASH2, 96)
                continue
            if steps < 16:                        # the two 1-beat inserts: band hit then silence
                hit(t_of(b, 0), KICK, 110)
                hit(t_of(b, 0), CRASH, 104)
                hit(t_of(b, 0), SNARE, 96)
                continue
            if k in ('intro_b', 'chorus', 'chorus2', 'vamp', 'vamp2', 'final', 'ending', 'reprise', 'build'):
                skip = set()
                if k == 'final' and b == 129:     # stop bar: only Rhodes + vocal, pickup into 130
                    for s, v in [(4, 46), (5, 56), (6, 70), (7, 86)]:
                        hit(t_of(b, s), SNARE, v)
                    groove(b + 1, 8, 'disco')
                    hit(t_of(b, 8), CRASH, 112)
                    hit(t_of(b, 8), CRASH2, 100)
                    continue
                if k == 'build' and last:         # roll that drops into silence at 129
                    for s in range(16):
                        hit(t_of(b, s), SNARE, 52 + 3.2 * s)
                    for s in (0, 4, 8, 12):
                        hit(t_of(b, s), KICK, 100)
                    continue
                if k == 'ending' and last:        # unison hits with the bass walk, band stops on 151
                    groove(b, 12, 'disco')
                    for s in (12, 13, 14, 15):
                        hit(t_of(b, s), SNARE, 90 + 4 * (s - 12))
                        hit(t_of(b, s), KICK, 96)
                    continue
                if last and k in ('intro_b', 'vamp', 'vamp2', 'reprise', 'chorus', 'chorus2'):
                    skip = fill(b, 'toms')
                elif k in ('chorus', 'chorus2', 'final', 'reprise') and idx % 2 == 1:
                    skip = fill(b, 'snare')
                vs = 0.88 if k == 'build' and idx < 2 else 1.0
                groove(b, steps, 'disco', skip=skip, vscale=vs)
                if k in ('final', 'reprise', 'chorus2'):
                    for s in (4, 12):
                        if s not in skip:
                            hit(t_of(b, s), TAMB, 70)
                if idx % 2 == 0 and not (k == 'build' and idx == 0):
                    hit(t_of(b, 0), CRASH if idx % 4 == 0 else CRASH2, 104 if idx == 0 else 88)
                elif k == 'build' and idx == 0:
                    hit(t_of(b, 0), CRASH, 104)
                continue
            if k == 'verse1':
                skip = fill(b, 'snare') if last else set()
                groove(b, steps, 'verse1', skip=skip, vscale=0.92 if idx < 4 else 1.0)
                continue
            if k == 'verse2':
                skip = fill(b, 'snare') if last else set()
                groove(b, steps, 'verse2', skip=skip)
                if idx % 2 == 1 and not last:
                    hit(t_of(b, 14), HHO, 58)
                if idx == 0:
                    hit(t_of(b, 0), CRASH2, 80)
                continue
            if k in ('pre1', 'pre2'):
                stop_step = 8 if k == 'pre2' and last else None
                if idx == 0:
                    hit(t_of(b, 0), CRASH2, 86)
                if last and stop_step is not None:   # pre-chorus 2: hit on beat 3 then silence
                    groove(b, 8, 'pre')
                    for s, v in [(4, 80), (5, 84), (6, 90), (7, 98)]:
                        hit(t_of(b, s), SNARE, v)
                    hit(t_of(b, 8), KICK, 110)
                    hit(t_of(b, 8), CRASH, 108)
                    continue
                if last:
                    groove(b, 8, 'pre')
                    for s in range(8, 16):
                        hit(t_of(b, s), SNARE, 62 + 5 * (s - 8))
                    hit(t_of(b, 14), TOM_L, 96)
                    hit(t_of(b, 15), TOM_FL, 104)
                    continue
                groove(b, steps, 'pre')
                if idx == count - 2:
                    for s in range(1, 16, 2):
                        hit(t_of(b, s), HHC, 40)
                continue
            if k == 'bridge':
                if b < 113:
                    groove(b, steps, 'bridge_a')
                else:
                    ramp = (b - 113) / 8.0
                    if last:
                        for s in range(16):
                            if s < 8 and s % 2:
                                continue
                            hit(t_of(b, s), SNARE, 50 + 3.0 * s)
                        hit(t_of(b, 0), KICK, 96)
                        hit(t_of(b, 8), KICK, 96)
                    else:
                        groove(b, steps, 'bridge_b', vscale=0.8 + 0.3 * ramp)
                continue
    return list(parts.values())


# ===========================================================================
# BASS
# ===========================================================================
DISCO = [
    [(0, 'R', 1.6, 102), (2, 'O', 0.8, 84), (3, 'R', 0.8, 76), (4, 'R', 1.4, 94), (6, 'O', 0.8, 86), (7, 'X', 0.8, 76)],
    [(0, 'R', 1.4, 102), (2, 'O', 0.8, 86), (3, 'O', 0.6, 70), (4, 'R', 1.8, 94), (6, 'O', 0.9, 84), (7, 'X', 0.8, 76)],
    [(0, 'R', 2.6, 102), (3, 'R', 0.8, 78), (4, 'O', 1.4, 88), (6, 'R', 0.8, 80), (7, 'X', 0.8, 76)],
    [(0, 'R', 1.4, 102), (2, 'O', 0.8, 84), (3, 'R', 0.8, 74), (4, 'F', 1.4, 88), (6, 'O', 0.8, 86), (7, 'X', 0.8, 76)],
]
VERSE = [(0, 'R', 2.6, 98), (3, 'R', 0.8, 74), (6, 'O', 0.9, 82), (7, 'X', 0.8, 72)]
VERSE2 = [(0, 'R', 1.8, 100), (3, 'R', 0.8, 76), (4, 'O', 0.9, 86), (6, 'R', 0.8, 80), (7, 'X', 0.8, 74)]
LINE = [(0, 'R', 5.2, 98), (6, 'O', 0.8, 76), (7, 'X', 0.8, 74)]
BUILD8 = [(0, 'R', 0.9, 98), (2, 'O', 0.9, 88), (4, 'R', 0.9, 96), (6, 'O', 0.9, 88)]
EIGHTHS = [(0, 'R', 0.9, 90), (2, 'R', 0.9, 74), (4, 'R', 0.9, 84), (6, 'O', 0.9, 82)]


def bass():
    part = Part('bass', program=33)
    prev = None

    def place(b, s, pat, chord, nxt, vscale=1.0, limit=8):
        nonlocal prev
        r = bass_pitch(chord.bass, prev)
        prev = r
        nr = bass_pitch(nxt.bass, r) if (nxt is not None and not nxt.is_nc) else None
        for st, what, dur, v in pat:
            if st >= limit:
                continue
            if what == 'R':
                p = r
            elif what == 'O':
                p = r + 12
            elif what == 'F':
                p = r + 7 if (chord.root is not None and (chord.root + 7) % 12 in chord.tone_pcs()) else r + 12
            else:   # 'X': approach note into the next chord, otherwise octave
                if nr is not None and nr != r:
                    p = nr - 1 if nr > r else nr + 1
                    if RNG.random() < 0.35:
                        p = nr + 7 - 12 if nr + 7 - 12 > 26 else r + 12
                else:
                    p = r + 12
            d = min(dur, limit - st) * STEP * 0.95
            part.add(t_of(b, s + st) + 0.003 + jitter(0.003), d, p, vel(v * vscale, 3))

    for sec in SECTIONS:
        k = kind_of(sec)
        for b, steps, idx, count in new_bars(sec):
            if k in ('pre', 'tag'):
                if k == 'tag' and idx == 0:     # E at the band hit of bar 151
                    part.add(t_of(b, 0), 6 * STEP, 28, 100)
                continue
            if k == 'end':
                if idx == 0:
                    part.add(t_of(b, 0), 20 * STEP, 33, 104)
                continue
            if steps < 16:                       # inserts: one short hit
                c = chord_at(b - 0.01)
                part.add(t_of(b, 0), 1.2 * STEP, bass_pitch(c.bass, prev), 104)
                continue
            for s0 in range(0, steps, 8):
                bar = b + s0 / 8
                for sb, se, c in spans(bar, bar + 1):
                    if c.is_nc:
                        continue
                    off = int(round((sb - b) * 8))
                    lim = int(round((se - sb) * 8))
                    nxt = chord_at(se + 0.001)
                    if k == 'intro_v':
                        if bar >= 4:
                            part.add(t_of(b, off), lim * STEP * 0.98, bass_pitch(c.bass, prev), 74)
                        continue
                    if k == 'final' and bar < 130:
                        continue
                    if k == 'pre2' and bar >= 84:
                        part.add(t_of(b, off), 1.2 * STEP, bass_pitch(c.bass, prev), 104)
                        continue
                    if k in ('intro_b', 'chorus', 'chorus2', 'vamp', 'vamp2', 'final', 'reprise'):
                        pat = DISCO[(int(bar * 2) + idx) % len(DISCO)]
                        place(b, off, pat, c, nxt, limit=lim)
                    elif k == 'ending':
                        if bar >= 150.5:
                            for i, p in enumerate([33, 32, 30, 28]):
                                part.add(t_of(b, off + i), 0.9 * STEP, p, 96 + 3 * i)
                            break
                        place(b, off, DISCO[int(bar * 2) % 4], c, nxt, limit=lim)
                    elif k == 'build':
                        place(b, off, BUILD8, c, nxt, vscale=0.9 + 0.04 * idx, limit=lim)
                    elif k == 'verse1':
                        if bar < 24.5:
                            place(b, off, [(0, 'R', 5.5, 90), (6, 'X', 1.6, 70)], c, nxt, limit=lim)
                        else:
                            place(b, off, VERSE, c, nxt, limit=lim)
                    elif k == 'verse2':
                        place(b, off, VERSE2, c, nxt, limit=lim)
                    elif k in ('pre1', 'pre2'):
                        place(b, off, LINE if lim >= 8 else [(0, 'R', 2.8, 96), (3, 'X', 0.8, 74)], c, nxt, limit=lim)
                    elif k == 'bridge':
                        if bar < 113:
                            if (bar - 105) % 2 == 0:
                                part.add(t_of(b, off), 15.5 * STEP, 33, 76)
                        else:
                            place(b, off, EIGHTHS, c, nxt, vscale=0.86 + 0.03 * (bar - 113), limit=lim)
    return [part]


# ===========================================================================
# RHODES
# ===========================================================================
COMP = [
    [(0, 2.0, 90), (3, 1.0, 68), (6, 2.0, 82), (10, 1.0, 70), (11, 3.0, 84)],
    [(2, 1.0, 78), (4, 2.0, 88), (7, 1.0, 72), (10, 2.0, 82), (14, 2.0, 80)],
    [(0, 3.0, 92), (6, 1.0, 70), (8, 2.0, 84), (11, 1.0, 68), (14, 2.0, 82)],
    [(2, 2.0, 80), (6, 2.0, 82), (10, 2.0, 80), (14, 2.0, 84)],
]
VCOMP = [
    [(0, 6.0, 74), (7, 1.0, 58), (10, 5.0, 68)],
    [(0, 4.0, 72), (6, 2.0, 60), (11, 4.0, 66)],
]


def roll(part, t, notes, d, v, spread=0.018, sd=0.004):
    for i, p in enumerate(notes):
        part.add(t + i * spread + jitter(sd), d - i * spread, p, vel(v - 2 * i, 3))


def comp_hits(part, voicer, b, steps, pattern, vscale=1.0, late=0.007, allow_push=True):
    for st, dur, v in pattern:
        if st >= steps:
            continue
        bar = b + st / 8
        c = chord_at(bar)
        nb = b + (st + 1) / 8
        if allow_push and st % 8 == 7 and chord_at(nb).symbol != c.symbol:
            c = chord_at(nb)               # anticipation ("push") of the next chord
            ce = next_change(nb)
        else:
            ce = next_change(bar)
        if c.is_nc:
            continue
        dmax = (ce - bar) * 8
        d = min(dur, dmax if dmax > 0.5 else dur) * STEP * 0.92
        notes = voicer(c)
        for i, p in enumerate(notes):
            part.add(t_of(b, st) + late + i * 0.004 + jitter(0.004), d, p, vel(v * vscale, 4))


def rhodes():
    part = Part('rhodes', program=4, bank=0)
    vo = Voicer('keys', lo=52, hi=76, center=63)
    for sec in SECTIONS:
        k = kind_of(sec)
        for b, steps, idx, count in new_bars(sec):
            if k in ('pre', 'intro_v', 'tag', 'bridge', 'end') or (k == 'final' and b == 129):
                lim = b + steps / 8
                for sb, se, c in spans(b, lim):
                    if c.is_nc:
                        continue
                    if k == 'final' and sb >= 130:
                        continue
                    d = (se - sb) * 8 * STEP + (0.25 if k != 'end' else 3.5)
                    v = {'pre': 66, 'intro_v': 64, 'tag': 62, 'bridge': 60, 'end': 70}.get(k, 70)
                    roll(part, t_of(sb) + 0.004, vo(c), d, v, spread=0.025 if k in ('pre', 'tag', 'end') else 0.016)
                    if k in ('intro_v', 'bridge') and se - sb >= 2:   # soft re-strike in long chords
                        roll(part, t_of(sb + 1, 4), vo(c), (se - sb - 1.5) * 8 * STEP, v - 12, spread=0.012)
                if k == 'final' and b == 129:
                    comp_hits(part, vo, b + 1, 8, [(0, 2.0, 94), (3, 1.0, 70), (6, 2.0, 84)])
                continue
            if steps < 16:
                c = chord_at(b - 0.01)
                roll(part, t_of(b), vo(c), 1.0 * STEP, 92, spread=0.004)
                continue
            if k in ('intro_b', 'chorus', 'chorus2', 'vamp', 'vamp2', 'final', 'reprise', 'ending'):
                pat = COMP[(idx + (1 if k in ('chorus2', 'reprise') else 0)) % len(COMP)]
                if k == 'ending' and idx == count - 1:
                    pat = [(0, 2.0, 92), (3, 1.0, 70), (6, 2.0, 84), (10, 1.0, 72)]
                comp_hits(part, vo, b, steps, pat)
            elif k in ('verse1', 'verse2'):
                comp_hits(part, vo, b, steps, VCOMP[idx % 2] if k == 'verse1' else COMP[(idx + 3) % 4],
                          vscale=0.9 if k == 'verse1' else 0.85)
            elif k in ('pre1', 'pre2', 'build'):
                stop = 8 if (k == 'pre2' and idx == count - 1) else steps
                pulses = [(s, 1.7, 56 + (idx * 16 + s) * (0.6 if k != 'build' else 0.9)) for s in range(0, stop, 2)]
                comp_hits(part, vo, b, stop, pulses, allow_push=False)
                if k == 'pre2' and idx == count - 1:
                    comp_hits(part, vo, b, 9, [(8, 1.0, 96)], allow_push=False)
    return [part]


# ===========================================================================
# FUNK GUITAR (cutting with upper-structure triads)
# ===========================================================================
def guitar():
    acc = Part('guitar', program=28, bank=8)        # MuseScore General "Funk Guitar"
    vo = Voicer('gtr', lo=62, hi=81, max_span=12, center=71)

    def strum(t, notes, d, v, up=False):
        seq = list(reversed(notes)) if up else notes
        for i, p in enumerate(seq):
            acc.add(t + i * 0.007 + jitter(0.002), d, p, vel(v - 3 * i, 3))

    for sec in SECTIONS:
        k = kind_of(sec)
        for b, steps, idx, count in new_bars(sec):
            if k in ('intro_b', 'chorus', 'chorus2', 'vamp', 'vamp2', 'final', 'reprise', 'ending', 'verse2', 'build') \
                    or (k == 'verse1' and b >= 24.5):
                if steps < 16 or (k == 'final' and b == 129):
                    continue
                if k in ('verse1', 'verse2'):
                    hits = [(4, 0.9, 84), (12, 0.9, 88)] + ([(10, 0.6, 62)] if idx % 2 else [])
                elif k == 'build':
                    hits = [(s, 0.7, 70 + 2 * s) for s in range(2, 16, 4)]
                else:
                    hits = [(2, 0.8, 90), (6, 0.8, 86), (10, 0.8, 90), (14, 0.8, 88)]
                    if idx % 2 == 1:
                        hits += [(7, 0.5, 70), (11, 0.5, 66)]
                    ghosts = [s for s in range(16) if s % 2 == 1 and s not in (7, 11)]
                    for s in ghosts:
                        c = chord_at(b + s / 8)
                        if not c.is_nc:
                            v = vo(c)
                            strum(t_of(b, s) + 0.004, v[:2], 0.045, 30, up=True)
                if k == 'ending' and idx == count - 1:
                    hits = [h for h in hits if h[0] < 12]
                for s, dur, v in hits:
                    bar = b + s / 8
                    c = chord_at(bar)
                    if s % 8 == 7 and chord_at(b + (s + 1) / 8).symbol != c.symbol:
                        c = chord_at(b + (s + 1) / 8)
                    if c.is_nc:
                        continue
                    strum(t_of(b, s) + 0.004, vo(c), dur * STEP, v, up=(s % 4 == 2))
            elif k in ('pre1', 'pre2'):
                stop = 8 if (k == 'pre2' and idx == count - 1) else steps
                order = [0, 1, 2, 1, 2, 0, 1, 2]
                for s in range(stop):
                    c = chord_at(b + s / 8)
                    if c.is_nc:
                        continue
                    v = vo(c)
                    p = v[order[s % 8] % len(v)]
                    acc.add(t_of(b, s) + 0.006 + jitter(0.003), 1.6 * STEP, p, vel(56 + 1.2 * (idx * 16 + s) / 4, 3))
    return [acc]


# ===========================================================================
# STRINGS, PAD, CHOIR (sustained parts with expression)
# ===========================================================================
def sustained(part, voicer, b0, b1, v, gap=0.02, tail=0.15):
    for sb, se, c in spans(b0, b1):
        if c.is_nc:
            continue
        notes = voicer(c)
        for p in notes:
            part.add(t_of(sb) + 0.01 + jitter(0.006), (se - sb) * 8 * STEP - gap + tail, p, vel(v, 2))


EXPR_CC = 2   # MuseScore General "Expr." presets read dynamics from CC2 (breath)


def expression_ramp(part, b0, b1, v0, v1, n=24):
    for i in range(n + 1):
        x = i / n
        part.ctrl(t_of(b0 + (b1 - b0) * x), EXPR_CC, v0 + (v1 - v0) * x)


def strings():
    st = Part('strings', program=49, bank=17)            # Strings Slow (expression)
    vio = Part('violins', program=48, bank=20)           # Violins Fast for runs / counter-line
    hi_v = Voicer('keys', lo=64, hi=88, max_span=19, center=76)   # violins/violas above the vocal
    low_prev = [None]

    def cello(c):
        pc = c.pcs('keys')[0]                             # guide tone (3rd / 4th of the chord)
        cands = [p for p in range(46, 58) if p % 12 == pc]
        ref = low_prev[0] if low_prev[0] is not None else 52
        p = min(cands, key=lambda q: abs(q - ref))
        low_prev[0] = p
        return p

    class _V:
        def __call__(self, c):
            if c.is_nc:
                return []
            return [cello(c)] + hi_v(c)

    vo = _V()
    st.ctrl(0.0, EXPR_CC, 80)
    plan = {
        'intro_v': (4, None, 46, 60), 'intro_b': (None, None, 70, 78), 'pre1': (None, None, 46, 112),
        'chorus': (None, None, 84, 88), 'pre2': (None, None, 50, 116), 'chorus2': (None, None, 88, 92),
        'bridge': (113, None, 36, 96), 'build': (None, None, 84, 124), 'final': (130, None, 96, 100),
        'ending': (None, None, 96, 90), 'tag': (None, None, 56, 44), 'reprise': (None, None, 86, 92),
        'end': (None, None, 90, 0),
    }
    for sec in SECTIONS:
        k = kind_of(sec)
        if k not in plan:
            continue
        s0, s1, e0, e1 = plan[k]
        b0 = sec['start'] if s0 is None else s0
        b1 = sec['end'] if s1 is None else s1
        if k == 'pre2':
            b1 = 84.25
        if k == 'pre1':
            b1 = 40.5
        expression_ramp(st, b0, b1, e0, e1, n=16)
        sustained(st, vo, b0, b1, 82 if k not in ('end',) else 88)
    # string run (32nds) into the band entry of the final chorus, and into the reprise
    for start, notes in [(129.5, [69, 71, 72, 74, 76, 77, 79, 81]), (154.5, [64, 67, 69, 71, 72, 74, 76, 79])]:
        for i, p in enumerate(notes):
            vio.add(t_of(start) + i * STEP / 2, STEP / 2 * 0.95, p, 70 + 5 * i)
            vio.add(t_of(start) + i * STEP / 2, STEP / 2 * 0.95, p + 12, 62 + 5 * i)
    # final chorus counter-line (violins, octaves) above the vocal
    prev = None
    for sb, se, c in spans(133, 145):
        if c.is_nc:
            continue
        cands = [p for p in range(81, 94) if p % 12 in c.tone_pcs()]
        p = min(cands, key=lambda q: (abs(q - prev) if prev else abs(q - 88)) + (0 if prev is None or q <= prev else 0.6))
        prev = p
        d = (se - sb) * 8 * STEP
        vio.add(t_of(sb) + 0.02, d * 0.96, p, 72)
        vio.add(t_of(sb) + 0.02, d * 0.96, p - 12, 64)
    return [st, vio]


def pad():
    part = Part('pad', program=89)                         # Warm Pad
    vo = Voicer('keys', lo=55, hi=77, center=66)
    for sec in SECTIONS:
        k = kind_of(sec)
        if k in ('pre', 'intro_v', 'verse1', 'verse2', 'bridge', 'tag', 'end'):
            b1 = sec['end']
            sustained(part, vo, sec['start'], b1, {'verse2': 52, 'verse1': 56}.get(k, 60), tail=0.6)
        if k in ('pre1', 'pre2'):
            sustained(part, vo, sec['start'], 40.5 if k == 'pre1' else 84.25, 54, tail=0.3)
    return [part]


def choir():
    part = Part('choir', program=53)                       # Voice Oohs (above the lead vocal)
    vo = Voicer('keys', lo=67, hi=86, center=76)
    for sec in SECTIONS:
        if kind_of(sec) in ('chorus2', 'final', 'ending'):
            b0 = 130 if kind_of(sec) == 'final' else sec['start']
            sustained(part, vo, b0, sec['end'], 58, tail=0.2)
    return [part]


# ===========================================================================
# BRASS: hook riff (from the original riff's 3+3+2 push and G-G#-A motif) and stabs
# ===========================================================================
HOOK_INTRO = [   # (step from 8.0, midi, dur steps) over Fmaj9 G13sus4 | Cadd9/E Am9 | Fmaj9 G13 | E7sus4 E7#9
    (0, 69, 2), (3, 69, 2), (6, 72, 2), (8, 74, 1), (10, 76, 1), (11, 74, 1), (12, 72, 2), (14, 67, 1), (15, 68, 1),
    (16, 69, 2), (19, 69, 2), (22, 72, 2), (24, 76, 3), (27, 74, 1), (28, 72, 1), (29, 69, 1), (30, 67, 2),
    (32, 69, 2), (35, 69, 2), (38, 72, 2), (40, 74, 1), (42, 76, 1), (43, 74, 1), (44, 77, 2), (46, 76, 1), (47, 74, 1),
    (48, 71, 2), (51, 71, 2), (54, 74, 2), (56, 79, 1), (57, 77, 1), (58, 76, 1), (59, 74, 1), (60, 71, 2), (62, 68, 2),
]
HOOK_VAMP = [    # over Am9 D9 | Am9 D9 (dorian vamp)
    (0, 69, 2), (3, 69, 2), (6, 72, 2), (8, 74, 1), (10, 76, 1), (11, 74, 1), (12, 72, 2), (14, 69, 1), (15, 71, 1),
    (16, 69, 2), (19, 69, 2), (22, 72, 2), (24, 78, 2), (26, 76, 1), (27, 74, 1), (28, 72, 2), (30, 69, 2),
]


def brass():
    part = Part('brass', program=61)                       # Brass Section
    sb_ = Part('synth_brass', program=62)                  # Synth Brass 1 (80s layer)
    hook = Part('hook', program=61)                        # melodic horn line (intro / interludes)
    vo = Voicer('keys', lo=62, hi=86, center=74)

    def line(b, notes, v=96):
        for s, p, d in notes:
            t = t_of(b, s) + 0.006 + jitter(0.003)
            for q, vv in ((p, v), (p - 12, v - 10)):
                hook.add(t, d * STEP * 0.9, q, vel(vv, 3))
            sb_.add(t, d * STEP * 0.9, p, vel(v - 18, 3))

    def stab(b, s, d, v=100):
        bar = b + s / 8
        c = chord_at(bar)
        if s % 8 == 7 and chord_at(b + (s + 1) / 8).symbol != c.symbol:
            c = chord_at(b + (s + 1) / 8)
        if c.is_nc:
            return
        for i, p in enumerate(vo(c)):
            part.add(t_of(b, s) + 0.005 + i * 0.003, d * STEP * 0.85, p, vel(v - 2 * i, 3))
            sb_.add(t_of(b, s) + 0.005, d * STEP * 0.85, p, vel(v - 24, 3))

    line(8, HOOK_INTRO)
    stab(16, 0, 1.0, 106)                                  # band hit before verse 1
    for start in (57, 101):
        line(start, HOOK_VAMP, 92)
    stab(40.5, 0, 1.0, 104)
    stab(84, 0, 1.0, 104)
    # chorus phrase-end stabs ("bap ... baaap" pushing into the next phrase)
    for c0 in (41, 85, 129, 155):
        for ph in range(4):
            b = c0 + 4 * ph + 2
            if c0 == 129 and ph == 3:
                continue
            stab(b, 11, 1.0, 96)
            stab(b, 14, 2.0, 104)
    stab(130, 0, 2.0, 110)                                 # band entry of the final chorus
    for b in (141, 143):
        stab(b, 0, 3.0, 100)
        stab(b, 6, 1.0, 92)
    stab(151, 0, 2.0, 110)                                 # band stop before the vocal tag
    # crescendo swell at the end of the build
    sw = Part('brass_swell', program=61, bank=17)
    for p in vo(chord_at(127.5)):
        sw.add(t_of(127), 8 * STEP * 2 - 0.03, p, 96)
    for i in range(25):
        sw.ctrl(t_of(127 + 2 * i / 24), EXPR_CC, 20 + 105 * (i / 24) ** 1.6)
    # final chord
    for p in vo(chord_at(171)):
        part.add(t_of(171) + 0.01, 1.4, p, 100)
    return [part, sb_, sw, hook]


# ===========================================================================
# MALLETS: vibraphone colours and glockenspiel sparkle
# ===========================================================================
def mallets():
    vib = Part('vibes', program=11)
    glk = Part('glock', program=9)
    vo = Voicer('keys', lo=72, hi=88, center=79)
    order = [0, 1, 2, 3, 2, 1, 0, 2]
    for sec in SECTIONS:
        k = kind_of(sec)
        for b, steps, idx, count in new_bars(sec):
            if k in ('intro_v', 'bridge', 'end') or (k == 'pre' and idx == 1):
                if k == 'bridge' and b >= 113:
                    continue
                for s in range(0, steps if k != 'end' else 16, 2):
                    c = chord_at(b + s / 8)
                    if c.is_nc:
                        c = chord_at(171)
                    v = vo(c)
                    vib.add(t_of(b, s) + 0.01 + jitter(0.004), 2.6 * STEP, v[order[(s // 2) % 8] % len(v)],
                            vel(50 if k != 'end' else 58 - s, 3))
            elif k == 'verse1' and idx % 2 == 1:
                c = chord_at(b + 1.5)
                v = vo(c)
                for i, s in enumerate((12, 13, 14, 15)):
                    vib.add(t_of(b, s) + 0.01, 1.6 * STEP, sorted(v, reverse=True)[i % len(v)], vel(52 - 2 * i, 3))
            elif k in ('chorus', 'chorus2', 'final', 'reprise') and idx % 2 == 0:
                if k == 'final' and b == 129:
                    continue
                c = chord_at(b)
                top = max(vo(c))
                glk.add(t_of(b) + 0.01, 3 * STEP, top + 12 if top + 12 <= 100 else top, 60)
    return [vib, glk]


# ===========================================================================
# LEAD (synth) - instrumental reprise of the chorus melody (from the vocal)
# ===========================================================================
DIATONIC = {0, 2, 4, 5, 7, 9, 11}       # A natural minor


def _snap(pf, chord):
    allowed = set(DIATONIC)
    if not chord.is_nc and chord.root == 4 and 4 in chord.q['tones']:
        allowed.add(8)                    # G# only over E (major-third) dominants
    cands = [p for p in range(int(pf) - 2, int(pf) + 3) if p % 12 in allowed]
    return min(cands, key=lambda p: abs(p - pf))


def lead_from_vocal(vocal_notes, src0=40.5, src1=57.0, dst0=154.5, octave=12):
    """Turn a section of the sung melody into a clean, legato synth line.

    The pYIN notes are syllabic and contain scoops; for an in-tune synth they are
    snapped to the key (G# allowed only over E chords), short slide fragments are
    dropped, repeated pitches are merged and the line is played legato.
    """
    from timeline import BAR, DB0
    raw = []
    for n in vocal_notes:
        bar = (n['t'] - DB0) / BAR
        if not (src0 <= bar < src1) or n['dur'] < 0.09:
            continue
        q = round((bar - src0) * 16) / 16                       # original 16th grid
        b = dst0 + q
        raw.append([b, n['beats'] / 4, _snap(n['pitch'], chord_at(b + 0.01)) + octave])
    raw.sort()
    merged = []
    for b, dur, p in raw:
        if merged and abs(merged[-1][0] - b) < 1e-6:
            if dur > merged[-1][1]:
                merged[-1] = [b, dur, p]
            continue
        if merged and merged[-1][2] == p and b - (merged[-1][0] + merged[-1][1]) < 0.12:
            merged[-1][1] = b + dur - merged[-1][0]               # tie repeated notes
            continue
        merged.append([b, dur, p])
    out = []
    for i, (b, dur, p) in enumerate(merged):
        nxt = merged[i + 1][0] if i + 1 < len(merged) else b + dur
        gap = nxt - (b + dur)
        if gap < 0.2:                                             # legato: run into the next note
            dur = nxt - b
        dur = min(dur, nxt - b) if i + 1 < len(merged) else dur
        out.append(Note(t_of(b), max(dur * BAR - 0.012, 0.08), p, 100))
    if dst0 == 154.5:
        out.append(Note(t_of(171) + 0.01, 2.8, 76, 92))          # sustained E5 over the final Am9
    return out


def all_parts():
    ps = []
    for fn in (drums, bass, rhodes, guitar, strings, pad, choir, brass, mallets):
        ps.extend(fn())
    return ps


if __name__ == '__main__':
    ps = all_parts()
    for p in ps:
        print(f'{p.name:12s} notes {len(p.notes):5d}  cc {len(p.cc):4d}')
