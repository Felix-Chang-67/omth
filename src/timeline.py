"""Tempo grid of the original recording and the time map of the rearrangement.

All arrangement data is written in *original bars* (191.016 BPM, 4/4), because the
original vocal stem is used without any time-stretching.  The new groove is a
half-time feel at 95.508 BPM, so one new bar = two original bars and one new
16th note = one original 8th note.

Values were measured from the drum stem (811 onsets, residual drift < 0.5 ms
across the whole song).
"""

T = 0.3141097            # one original beat in seconds (191.016 BPM)
PHASE = 0.032            # original-audio time of beat index 0 (drum attack)
BAR = 4 * T              # one original bar
DB0 = PHASE + T          # downbeat of original bar 0 (the first full bar)
STEP = T / 2             # one 16th note of the new half-time groove
NEW_BEAT = 2 * T         # one beat of the new 95.508 BPM groove
NEW_BPM = 60.0 / NEW_BEAT

PRE_ROLL_BARS = 4        # new Rhodes intro before the vocal: 4 original bars = 2 new bars
OFFSET = PRE_ROLL_BARS * BAR   # output time = original time + OFFSET

SR = 44100


def orig_time(bar: float) -> float:
    """Time (s) in the original stems of original bar `bar` (fractional allowed)."""
    return DB0 + bar * BAR


def out_time(bar: float) -> float:
    """Time (s) in the rendered arrangement of original bar `bar`."""
    return orig_time(bar) + OFFSET


def step_time(bar: float, step: float) -> float:
    """Output time of 16th-note `step` (new groove) counted from original bar `bar`."""
    return out_time(bar) + step * STEP
