"""Renders the generated parts with FluidSynth + custom synths, then mixes and masters."""
from __future__ import annotations

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import mido
import numpy as np
import pyloudnorm as pyln
import scipy.signal as ss
import soundfile as sf
from pedalboard import (Chorus, Compressor, Delay, Gain, HighShelfFilter, HighpassFilter, Limiter,
                        LowShelfFilter, LowpassFilter, PeakFilter, Pedalboard)

import synth
from parts import Part
from timeline import OFFSET, STEP, SR, out_time

SOUNDFONT = '/usr/share/sounds/sf3/MuseScore_General_Full.sf3'
METER = pyln.Meter(SR)


# ---------------------------------------------------------------------------
# MIDI + FluidSynth
# ---------------------------------------------------------------------------
def _fix_overlaps(notes):
    by_pitch = {}
    for n in notes:
        by_pitch.setdefault(n.p, []).append(n)
    out = []
    for p, ns in by_pitch.items():
        ns.sort(key=lambda n: n.t)
        for a, b in zip(ns, ns[1:]):
            if a.t + a.d > b.t - 0.002:
                a.d = max(0.01, b.t - a.t - 0.002)
        out.extend(ns)
    return out


def write_midi(part: Part, path: str, tail: float = 6.0, tempo_bpm: float = 120.0):
    mid = mido.MidiFile(ticks_per_beat=960)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tempo = mido.bpm2tempo(tempo_bpm)
    spt = tempo / 1e6 / 960
    tr.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    ch = 9 if part.drum else 0
    if not part.drum:
        tr.append(mido.Message('control_change', channel=ch, control=0, value=part.bank, time=0))
        tr.append(mido.Message('control_change', channel=ch, control=32, value=0, time=0))
    tr.append(mido.Message('program_change', channel=ch, program=part.program, time=0))
    ev = []
    for n in _fix_overlaps(list(part.notes)):
        ev.append((max(0.0, n.t), 2, 'on', n.p, n.v))
        ev.append((max(0.0, n.t + n.d), 0, 'off', n.p, 0))
    for t, c, v in part.cc:
        ev.append((max(0.0, t), 1, 'cc', c, v))
    ev.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for t, _, kind, a, b in ev:
        tick = int(round(t / spt))
        dt, last = tick - last, tick
        if kind == 'on':
            tr.append(mido.Message('note_on', channel=ch, note=a, velocity=b, time=dt))
        elif kind == 'off':
            tr.append(mido.Message('note_off', channel=ch, note=a, velocity=0, time=dt))
        else:
            tr.append(mido.Message('control_change', channel=ch, control=a, value=b, time=dt))
    tr.append(mido.MetaMessage('end_of_track', time=int(tail / spt)))
    mid.save(path)


def render_part(part: Part, out_wav: str):
    with tempfile.TemporaryDirectory() as td:
        mpath = os.path.join(td, 'part.mid')
        write_midi(part, mpath)
        cmd = ['fluidsynth', '-ni', '-q', '-g', '0.6', '-R', '0', '-C', '0', '-r', str(SR), '-O', 'float',
               '-T', 'wav', '-o', 'synth.polyphony=512', '-F', out_wav, SOUNDFONT, mpath]
        subprocess.run(cmd, check=True, capture_output=True)
    return out_wav


def load(path, n):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    assert sr == SR
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    out = np.zeros((n, 2), np.float32)
    m = min(n, len(x))
    out[:m] = x[:m]
    return out


# ---------------------------------------------------------------------------
# small DSP helpers
# ---------------------------------------------------------------------------
def fx(x, *plugins):
    return Pedalboard(list(plugins))(x.T.copy(), SR).T.astype(np.float32)


def pan(x, p):
    """Balance-style pan for stereo material, p in [-1, 1]."""
    gl = np.cos((p + 1) * np.pi / 4) * np.sqrt(2)
    gr = np.sin((p + 1) * np.pi / 4) * np.sqrt(2)
    y = x.copy()
    y[:, 0] *= min(gl, 1.0)
    y[:, 1] *= min(gr, 1.0)
    return y


def mono_to_wide(x, ms=0.012):
    """Haas-style widening of a (near) mono source."""
    m = x.mean(axis=1)
    d = int(ms * SR)
    l = m
    r = np.concatenate([np.zeros(d, np.float32), m[:-d]])
    return np.stack([l, 0.85 * r + 0.15 * m], axis=1).astype(np.float32)


def autopan(x, rate=4.2, depth=0.22):
    t = np.arange(len(x)) / SR
    lfo = np.sin(2 * np.pi * rate * t)
    y = x.copy()
    y[:, 0] *= 1 - depth * (0.5 + 0.5 * lfo)
    y[:, 1] *= 1 - depth * (0.5 - 0.5 * lfo)
    return y


def deess(x, lo=5200, hi=9500, thresh=0.35, depth=0.6):
    b, a = ss.butter(2, [lo / (SR / 2), hi / (SR / 2)], 'band')
    band = ss.lfilter(b, a, x, axis=0)
    def env(s):
        e = np.abs(s).mean(axis=1)
        return ss.lfilter([0.01], [1, -0.99], e) + 1e-7
    ratio = env(band) / env(x)
    g = np.where(ratio > thresh, 1 - depth * np.clip((ratio - thresh) / thresh, 0, 1), 1.0)
    return (x - band + band * g[:, None]).astype(np.float32)


def loudness(x):
    try:
        return METER.integrated_loudness(x.astype(np.float64))
    except Exception:
        return -70.0


def set_loudness(x, target):
    L = loudness(x)
    if not np.isfinite(L) or L < -69:
        return x, 0.0
    g = 10 ** ((target - L) / 20)
    return (x * g).astype(np.float32), target - L


def true_peak_db(x):
    up = ss.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


# ---------------------------------------------------------------------------
# per-stem processing chains: (fx function, pan, plate send, room send, target LUFS)
# ---------------------------------------------------------------------------
CHAINS = {
    'kick':       (lambda x: fx(x, HighpassFilter(28), PeakFilter(58, 3.0, 0.9), PeakFilter(330, -4.5, 1.2),
                                HighShelfFilter(4500, 2.0), Compressor(-14, 3.5, 8, 90)), 0.0, 0.00, 0.03, -21.5),
    'snare':      (lambda x: fx(x, HighpassFilter(110), PeakFilter(210, 2.0, 1.0), PeakFilter(5000, 2.0, 0.8),
                                Compressor(-16, 3.0, 4, 120)), 0.0, 0.26, 0.10, -24.0),
    'clap':       (lambda x: fx(x, HighpassFilter(320), Compressor(-18, 3.0, 2, 100)), 0.0, 0.32, 0.0, -31.0),
    'hats':       (lambda x: fx(x, HighpassFilter(380), HighShelfFilter(9000, 1.5)), 0.28, 0.04, 0.05, -30.5),
    'cymbals':    (lambda x: fx(x, HighpassFilter(320)), -0.1, 0.05, 0.08, -30.0),
    'toms':       (lambda x: fx(x, HighpassFilter(60), Compressor(-16, 3.0, 5, 120)), 0.0, 0.10, 0.12, -27.0),
    'perc':       (lambda x: fx(x, HighpassFilter(450)), -0.35, 0.06, 0.05, -34.0),
    'bass':       (lambda x: fx(x, HighpassFilter(32), Compressor(-18, 4.0, 6, 70), PeakFilter(90, 1.5, 1.0),
                                PeakFilter(250, -2.5, 1.0), PeakFilter(900, 2.5, 1.0), LowpassFilter(6000)), 0.0, 0.0, 0.0, -21.0),
    'rhodes':     (lambda x: autopan(fx(x, HighpassFilter(90), PeakFilter(280, -2.0, 1.0), Chorus(0.55, 0.22, 7.0, 0.0, 0.38),
                                        Compressor(-20, 2.0, 15, 160))), -0.12, 0.20, 0.0, -24.5),
    'guitar':     (lambda x: fx(x, HighpassFilter(200), Compressor(-20, 3.0, 4, 80), PeakFilter(2600, -2.5, 1.1),
                                PeakFilter(5500, 1.5, 1.0), Chorus(0.9, 0.12, 5.0, 0.0, 0.25)), 0.55, 0.10, 0.05, -29.5),
    'strings':    (lambda x: fx(x, HighpassFilter(110), LowShelfFilter(260, -2.5), PeakFilter(3000, -2.0, 1.0),
                                HighShelfFilter(8000, 1.5)), 0.0, 0.30, 0.0, -27.0),
    'violins':    (lambda x: fx(x, HighpassFilter(220), HighShelfFilter(7000, 1.0)), 0.2, 0.32, 0.0, -29.5),
    'pad':        (lambda x: fx(x, HighpassFilter(160), LowpassFilter(7000), Chorus(0.3, 0.3, 9.0, 0.0, 0.5)), 0.0, 0.35, 0.0, -31.0),
    'choir':      (lambda x: fx(x, HighpassFilter(220), PeakFilter(3500, -2.0, 1.0)), 0.0, 0.38, 0.0, -32.0),
    'brass':      (lambda x: fx(x, HighpassFilter(140), Compressor(-18, 2.5, 6, 100), PeakFilter(2800, -1.5, 1.0)), -0.22, 0.20, 0.06, -26.0),
    'hook':       (lambda x: fx(x, HighpassFilter(140), Compressor(-20, 3.0, 5, 90), PeakFilter(1800, 1.5, 1.0)), -0.15, 0.22, 0.06, -18.0),
    'synth_brass': (lambda x: fx(x, HighpassFilter(160), Chorus(0.4, 0.25, 8.0, 0.0, 0.4)), 0.25, 0.18, 0.0, -32.0),
    'brass_swell': (lambda x: fx(x, HighpassFilter(140)), -0.1, 0.22, 0.0, -27.0),
    'vibes':      (lambda x: fx(x, HighpassFilter(250)), -0.4, 0.30, 0.0, -30.0),
    'glock':      (lambda x: fx(x, HighpassFilter(500)), 0.4, 0.30, 0.0, -34.0),
}
GROUPS = {
    'drums': ['kick', 'snare', 'clap', 'hats', 'cymbals', 'toms', 'perc'],
    'bass': ['bass'],
    'keys': ['rhodes', 'vibes', 'glock'],
    'guitar': ['guitar'],
    'strings': ['strings', 'violins'],
    'brass': ['brass', 'synth_brass', 'brass_swell', 'hook'],
    'pads': ['pad', 'choir'],
}


def process_vocal(v):
    v = fx(v, HighpassFilter(95), PeakFilter(320, -2.0, 1.0), PeakFilter(660, -2.5, 1.3), Compressor(-22, 3.0, 4, 90),
           PeakFilter(3000, 2.0, 0.9), PeakFilter(9500, -2.5, 0.9), HighShelfFilter(14000, 1.0), Gain(0.0))
    v = deess(v, lo=5000, hi=11000, thresh=0.3, depth=0.5)
    return v


# section rides (dB) per stem, applied with short ramps: the verses sit lower than the choruses
RIDES = {
    'verse1': {'kick': -3, 'snare': -3, 'clap': -3, 'hats': -3, 'perc': -2, 'cymbals': -2, 'toms': -2,
               'bass': -1.5, 'rhodes': -1, 'guitar': -2},
    'verse2': {'kick': -1.5, 'snare': -1.5, 'hats': -1.5, 'perc': -1, 'bass': -0.5, 'guitar': -1},
    'bridge': {'kick': -1, 'snare': -1, 'hats': -1, 'perc': -1},
    'intro_vocal': {'bass': -2},
}


def ride_envelope(name, n):
    from chords import SECTIONS
    g = np.zeros(n, np.float32)
    for sec in SECTIONS:
        db = RIDES.get(sec['name'], {}).get(name, 0.0)
        if db:
            a, b = int(out_time(sec['start']) * SR), int(out_time(sec['end']) * SR)
            g[a:b] = db
    k = int(0.06 * SR)
    g = np.convolve(g, np.ones(k) / k, mode='same')
    return (10 ** (g / 20)).astype(np.float32)


# macro-dynamics: whole-mix section rides (dB) applied before mastering
MASTER_RIDES = {'pre_intro': -1.0, 'intro_vocal': -1.5, 'verse1': -1.5, 'verse2': -1.0, 'bridge': -2.0,
                'tag': -1.0, 'final_chorus': 0.5, 'ending': 0.5}


def master_ride(n):
    from chords import SECTIONS
    g = np.zeros(n, np.float32)
    for sec in SECTIONS:
        db = MASTER_RIDES.get(sec['name'], 0.0)
        a, b = int(out_time(sec['start']) * SR), int(out_time(sec['end']) * SR)
        g[a:b] = db
    k = int(0.08 * SR)
    g = np.convolve(g, np.ones(k) / k, mode='same')
    return (10 ** (g / 20)).astype(np.float32)[:, None]


def widen(x, w):
    m = 0.5 * (x[:, 0] + x[:, 1])
    s = 0.5 * (x[:, 0] - x[:, 1]) * w
    return np.stack([m + s, m - s], axis=1).astype(np.float32)


def mix(parts, vocal_raw, lead_notes, out_dir, total_s, work_dir, log=print):
    n = int(total_s * SR)
    os.makedirs(work_dir, exist_ok=True)
    # 1) render all SoundFont parts in parallel
    jobs = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for p in parts:
            jobs[p.name] = ex.submit(render_part, p, os.path.join(work_dir, f'{p.name}.wav'))
        paths = {k: f.result() for k, f in jobs.items()}
    log(f'rendered {len(paths)} SoundFont parts')
    raw = {k: load(v, n) for k, v in paths.items()}

    # 2) layer a synthesized sub under the GM kick
    kick_part = next(p for p in parts if p.name == 'kick')
    sub = np.zeros((n, 2), np.float32)
    for note in kick_part.notes:
        synth.place(sub, note.t, synth.sub_kick(note.v / 127), 1.0)
    raw['kick'] = raw['kick'] + 0.55 * sub * (np.abs(raw['kick']).max() / (np.abs(sub).max() + 1e-9))

    # 3) process stems, set loudness, collect sends
    plate_send = np.zeros((n, 2), np.float32)
    room_send = np.zeros((n, 2), np.float32)
    stems = {}
    for name, x in raw.items():
        chain, p, plate, room, target = CHAINS[name]
        y = chain(x)
        if name in ('vibes', 'glock', 'guitar', 'perc'):
            y = mono_to_wide(y, 0.009) if name == 'guitar' else y
        y = pan(y, p)
        if name in ('strings', 'pad', 'choir', 'violins'):
            y = widen(y, 1.6)
        elif name == 'rhodes':
            y = widen(y, 1.3)
        y, g = set_loudness(y, target)
        y = y * ride_envelope(name, n)[:, None]
        stems[name] = y
        plate_send += plate * y
        room_send += room * y
        log(f'  {name:12s} -> {target:6.1f} LUFS (gain {g:+5.1f} dB)')

    # 4) lead synth for the instrumental reprise
    if lead_notes:
        t0 = min(nt.t for nt in lead_notes) - 0.1
        t1 = max(nt.t + nt.d for nt in lead_notes) + 0.5
        ld = synth.lead_synth([(nt.t, nt.d, nt.p, nt.v) for nt in lead_notes], t0, t1)
        ld = np.stack([ld, ld], axis=1)
        lead = np.zeros((n, 2), np.float32)
        synth.place(lead, t0, ld)
        oct_up = synth.lead_synth([(nt.t, nt.d, nt.p + 12, nt.v) for nt in lead_notes], t0, t1)
        synth.place(lead, t0, 0.22 * np.stack([oct_up, oct_up], axis=1))
        lead = fx(lead, HighpassFilter(150), Chorus(0.8, 0.18, 6.0, 0.0, 0.3), PeakFilter(2800, 2.0, 1.0),
                  Compressor(-18, 2.5, 5, 120))
        dly = fx(lead, Delay(3 * STEP, 0.32, 1.0), HighpassFilter(400), LowpassFilter(5000))
        lead, g = set_loudness(lead, -16.5)
        stems['lead'] = lead + 0.20 * dly * 10 ** (g / 20)
        plate_send += 0.22 * stems['lead']
        log('  lead         -> -16.5 LUFS')

    # 5) transition FX
    fxbus = np.zeros((n, 2), np.float32)
    for bar, length in ((8, 1.6), (130, 1.5), (155, 1.5), (0, 2.2)):
        rc = synth.reverse_cymbal(length)
        synth.place(fxbus, out_time(bar) - len(rc) / SR, rc, 0.9 if bar else 0.5)
    for bar, dur in ((119, 2 * 2.513), (127, 2.513)):
        r = synth.riser(dur)
        synth.place(fxbus, out_time(bar), r, 0.7)
    for bar in (41, 85, 130, 155):
        synth.place(fxbus, out_time(bar), np.stack([synth.sub_drop()] * 2, axis=1), 0.35)
    vin = synth.vinyl(out_time(4) + 1.0)
    fade = np.linspace(1, 0, len(vin))[:, None] ** 0.7
    synth.place(fxbus, 0.0, vin * fade, 1.0)
    vin2 = synth.vinyl(total_s - out_time(151), seed=9)
    fin = np.linspace(0, 1, len(vin2))[:, None]
    synth.place(fxbus, out_time(151), vin2 * np.minimum(1, fin * 3) * 0.8, 1.0)
    stems['fx'] = fxbus
    plate_send += 0.15 * fxbus

    # 6) vocal
    voc = np.zeros((n, 2), np.float32)
    synth.place(voc, OFFSET, vocal_raw)
    voc = process_vocal(voc)
    voc, g = set_loudness(voc, -16.0)
    vdelay = fx(voc, Delay(3 * STEP, 0.30, 1.0), HighpassFilter(350), LowpassFilter(4500))
    stems['vocal'] = voc
    log(f'  vocal        -> -16.0 LUFS (gain {g:+5.1f} dB)')

    # 7) reverbs
    plate_ir = synth.make_ir((2.6, 2.3, 1.9, 1.2), predelay=0.025, seed=21)
    room_ir = synth.make_ir((0.9, 0.8, 0.6, 0.4), predelay=0.008, seed=22)
    vplate = synth.convolve(fx(voc, HighpassFilter(250), LowpassFilter(9000)), plate_ir)
    plate_ret = synth.convolve(fx(plate_send, HighpassFilter(200), LowpassFilter(10000)), plate_ir)
    room_ret = synth.convolve(fx(room_send, HighpassFilter(150)), room_ir)
    returns = 0.85 * plate_ret + 0.6 * room_ret
    vox_fx = 0.16 * vplate + 0.11 * vdelay
    stems['returns'] = returns
    stems['vocal_fx'] = vox_fx

    # 8) buses
    drums = sum(stems[k] for k in GROUPS['drums'])
    drums = fx(drums, Compressor(-12, 2.0, 15, 120))
    music = sum(stems[k] for k in stems if k not in GROUPS['drums'] + ['vocal', 'vocal_fx', 'returns', 'fx'])
    mixbus = drums + music + stems['fx'] + returns + stems['vocal'] + vox_fx
    return mixbus, stems, drums


def lookahead_limiter(x, ceiling_db=-1.0, lookahead_ms=2.0, release_ms=90.0):
    """Brick-wall limiter driven by the 4x oversampled (true) peak of the signal."""
    ceil = 10 ** (ceiling_db / 20)
    up = np.abs(ss.resample_poly(x, 4, 1, axis=0)).max(axis=1)
    pk = up[:len(up) // 4 * 4].reshape(-1, 4).max(axis=1)
    pk = np.concatenate([pk, np.zeros(len(x) - len(pk))])
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = max(1, int(lookahead_ms * SR / 1000))
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    hold = minimum_filter1d(need, size=2 * la + 1, origin=0)
    # release: gain may only rise slowly (one-pole), computed in a vectorised peak-hold style per block
    a = np.exp(-1.0 / (release_ms * SR / 1000))
    g = np.empty_like(hold)
    cur = 1.0
    for i0 in range(0, len(hold), 4096):
        blk = hold[i0:i0 + 4096]
        out = np.empty_like(blk)
        for j, h in enumerate(blk):
            cur = h if h < cur else h + (cur - h) * a
            out[j] = cur
        g[i0:i0 + 4096] = out
    g = uniform_filter1d(g, size=la, origin=0)
    g = np.minimum(g, hold)
    return (x * g[:, None]).astype(np.float32)


def master(mixbus, target_lufs=-12.0, ceiling_db=-1.5, log=print):
    y = fx(mixbus, HighpassFilter(24), LowShelfFilter(90, 0.8), HighShelfFilter(11000, 1.2),
           Compressor(-16, 1.8, 30, 250))
    gain = 0.0
    for _ in range(6):
        z = fx(y, Gain(gain), Limiter(ceiling_db + 0.5, 60))     # soft first stage
        z = lookahead_limiter(z, ceiling_db - 0.1)
        L = loudness(z)
        tp = true_peak_db(z)
        log(f'  master pass: gain {gain:+.2f} dB -> {L:.2f} LUFS, true peak {tp:.2f} dBTP')
        if abs(L - target_lufs) < 0.15:
            break
        gain += target_lufs - L
    return z.astype(np.float32), float(L), float(true_peak_db(z))
