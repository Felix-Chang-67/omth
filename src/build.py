"""Build the Midnight City Pop rearrangement of "One Meets Her".

usage: python3 build.py --stems <dir with the 5 separated stems> --out <output dir>

The stems folder must contain 04_one_meets_her_vocals.wav (the only original
stem used in the final mix; drums/bass/guitar are replaced by new parts).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import mido
import numpy as np
import soundfile as sf

import analysis
import parts as P
import render
from chords import SECTIONS
from timeline import NEW_BEAT, NEW_BPM, OFFSET, SR, out_time

TITLE = 'One Meets Her (Midnight City Pop Rearrangement)'


def write_arrangement_midi(parts, path, lead=None):
    """Multi-track MIDI on the new 95.5 BPM grid; tick 0 = first downbeat of the pre-intro."""
    t0 = out_time(-4)
    mid = mido.MidiFile(ticks_per_beat=960, type=1)
    spt = NEW_BEAT / 960
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage('track_name', name=TITLE, time=0))
    meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(NEW_BPM), time=0))
    ts = [(-4, 4), (14, 5), (16.5, 4), (38.5, 5), (41, 4)]
    last = 0
    for bar, num in ts:
        tick = int(round((out_time(bar) - t0) / spt))
        meta.append(mido.MetaMessage('time_signature', numerator=num, denominator=4, time=tick - last))
        last = tick
    for s in SECTIONS:
        tick = int(round((out_time(s['start']) - t0) / spt))
        meta.append(mido.MetaMessage('marker', text=s['name'], time=max(0, tick - last)))
        last = max(last, tick)
    mid.tracks.append(meta)
    allp = list(parts) + ([lead] if lead else [])
    for ch_i, p in enumerate(allp):
        ch = 9 if p.drum else [c for c in range(16) if c != 9][ch_i % 15]
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage('track_name', name=p.name, time=0))
        if not p.drum:
            tr.append(mido.Message('control_change', channel=ch, control=0, value=p.bank, time=0))
        tr.append(mido.Message('program_change', channel=ch, program=p.program, time=0))
        ev = []
        for n in render._fix_overlaps(list(p.notes)):
            ev.append((n.t, 1, 'on', n.p, n.v))
            ev.append((n.t + n.d, 0, 'off', n.p, 0))
        for t, c, v in p.cc:
            ev.append((t, 0.5, 'cc', c, v))
        ev.sort(key=lambda e: (e[0], e[1]))
        lt = 0
        for t, _, kind, a, b in ev:
            tick = max(0, int(round((t - t0) / spt)))
            dt, lt = tick - lt, tick
            if kind == 'on':
                tr.append(mido.Message('note_on', channel=ch, note=a, velocity=b, time=dt))
            elif kind == 'off':
                tr.append(mido.Message('note_off', channel=ch, note=a, velocity=0, time=dt))
            else:
                tr.append(mido.Message('control_change', channel=ch, control=a, value=b, time=dt))
        mid.tracks.append(tr)
    mid.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stems', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', default=None, help='scratch dir for intermediate renders')
    ap.add_argument('--midi-out', default=None, help='where to write the arrangement MIDI (default: <out>/midi)')
    ap.add_argument('--no-lead-in-midi', action='store_true')
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    work = args.work or os.path.join(args.out, '_work')

    vocal, sr = sf.read(os.path.join(args.stems, '04_one_meets_her_vocals.wav'), dtype='float32', always_2d=True)
    assert sr == SR
    print('extracting the vocal melody for the reprise lead ...')
    vnotes = analysis.vocal_notes(vocal, sr)

    print('generating parts ...')
    parts = P.all_parts()
    lead_notes = P.lead_from_vocal(vnotes)
    lead_part = P.Part('lead', program=81)
    lead_part.notes = list(lead_notes)
    for p in parts:
        print(f'  {p.name:12s} {len(p.notes):5d} notes')
    print(f'  lead         {len(lead_notes):5d} notes')

    total_s = out_time(171) + 6.5
    print(f'rendering + mixing ({total_s:.1f} s) ...')
    mixbus, stems, drums = render.mix(parts, vocal, lead_notes, args.out, total_s, work)
    ride = render.master_ride(len(mixbus))
    mixbus = mixbus * ride
    print('mastering ...')
    master, L, tp = render.master(mixbus)
    fade = int(1.5 * SR)
    master[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2

    base = os.path.join(args.out, 'one_meets_her_midnight_city_pop')
    sf.write(base + '.wav', master, SR, subtype='PCM_24')
    subprocess.run(['lame', '--quiet', '-b', '320', '--tt', TITLE, '--ta', 'rearranged with Claude Code',
                    base + '.wav', base + '.mp3'], check=True)
    # instrumental version (no vocal, no vocal fx) through the same master gain staging
    inst = mixbus - (stems['vocal'] + stems['vocal_fx']) * ride
    inst_m, _, _ = render.master(inst, target_lufs=L - 1.0, log=lambda *a: None)
    inst_m[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
    sf.write(base + '_instrumental.wav', inst_m, SR, subtype='PCM_24')
    subprocess.run(['lame', '--quiet', '-b', '320', '--tt', TITLE + ' [Instrumental]', base + '_instrumental.wav',
                    base + '_instrumental.mp3'], check=True)
    # grouped stems (pre-master, post-fader)
    sd = os.path.join(args.out, 'stems')
    os.makedirs(sd, exist_ok=True)
    groups = dict(render.GROUPS)
    groups.update({'lead': ['lead'], 'fx': ['fx'], 'reverb_returns': ['returns'], 'vocal': ['vocal', 'vocal_fx']})
    for g, names in groups.items():
        if g == 'drums':
            x = drums
        else:
            x = sum(stems[k] for k in names if k in stems)
        sf.write(os.path.join(sd, f'{g}.flac'), np.clip(x * 0.7, -1, 1), SR, subtype='PCM_24')
    md = args.midi_out or os.path.join(args.out, 'midi')
    os.makedirs(md, exist_ok=True)
    write_arrangement_midi(parts, os.path.join(md, 'one_meets_her_city_pop_arrangement.mid'),
                           None if args.no_lead_in_midi else lead_part)
    with open(os.path.join(args.out, 'render_report.json'), 'w') as f:
        json.dump(dict(integrated_lufs=round(float(L), 2), true_peak_dbtp=round(float(tp), 2),
                       duration_s=round(len(master) / SR, 2), offset_of_original_vocal_s=round(OFFSET, 4),
                       new_bpm=round(NEW_BPM, 3)), f, indent=1)
    print(f'done in {time.time() - t_start:.0f} s: {L:.2f} LUFS, {tp:.2f} dBTP -> {base}.wav/.mp3')


if __name__ == '__main__':
    main()
