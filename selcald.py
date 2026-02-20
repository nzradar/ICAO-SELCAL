import numpy as np
import sounddevice as sd
import json
import time
from datetime import datetime
from collections import deque

# ================= CONFIG =================

SAMPLE_RATE = 8000
INPUT_DEVICE = 1
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.2 # Was 0.08

FULL_CODE_LOCKOUT = 4.0
DECODE_LOCKOUT = 1.2 # Was 1.0
SILENCE_RMS_MAX = 0.00015 # Was 0.002 
SILENCE_RMS_MIN = 0.00015

PAIR_HIT_REQUIRED = 4
PAIR_ACCUM_WINDOW = 1.2
FLATNESS_MAX = 0.8 # Was 0.8

GOERTZEL_MIN_POWER = 0.00018 #Was 0.00018
GOERTZEL_RATIO = 1.8 # Was 1.4
PAIR_POWER_SUM = 0.003 # Was 0.0035
PAIR_POWER_MIN = 0.015   # tuneable, start here
PAIR_MIN_DURATION = 0.12   # seconds (0.4–0.6 works well) Was 0.45
PAIR_GAP_MIN = 0.45
PAIR_GAP_MAX = 0.9
PAIR_POWER_MIN = 0.15   # start here (HF); VHF can go higher
PAIR_IMBALANCE_MAX = 4.0   # ratio (4:1 is generous)

DEBUG = False

LOG_FILE = "selcal_log.txt"
DICT_FILE = r"C:\Scripts\selcal_decoder\SELCAL_DICTIONARY.TXT"
JSON_FILE = r"C:\Scripts\selcal_decoder\frequencies.json"

# ================= DEBUG =================

def dbg(msg):
    if DEBUG:
        print(msg)

# ================= SELCAL Code Validation ============== 

def is_valid_selcal(code):
    if len(code) != 4:
        return False

    a, b, c, d = code

    # no duplicates anywhere
    if len(set(code)) != 4:
        return False

    # pairs must be distinct internally (defensive)
    if a == b or c == d:
        return False

    return True


# ================= LOGGING =================

def log_selcal(line):
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            old = f.read()
    except FileNotFoundError:
        old = ""

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(line + "\n")
        f.write(old)

# ================= UTIL =================

def load_selcal_dictionary(path):
    d = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                code, reg, ac, op = parts[:4]
                d[code.strip()] = f"({reg} {ac} {op})"
    print(f"Loaded {len(d)} SELCAL entries")
    return d
    
# ================= Pair Tracker ========

class PairTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_pair = None
        self.start_time = None
        self.first_pair = None
        self.first_time = None

    def update(self, pair, now):
        # still waiting for first pair
        if self.first_pair is None:
            self.first_pair = pair
            self.first_time = now
            return None

        # same pair repeating → ignore
        if pair == self.first_pair:
            return None

        gap = now - self.first_time

        # enforce ICAO timing
        if not (PAIR_GAP_MIN <= gap <= PAIR_GAP_MAX):
            self.reset()
            return None

        # got full SELCAL
        code = self.first_pair + pair
        self.reset()
        return code




# ================= DSP =================

def goertzel_mag(samples, freq, fs):
    n = len(samples)
    k = int(0.5 + (n * freq) / fs)
    w = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(w)

    s_prev = s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    return np.sqrt(s_prev2**2 + s_prev**2 - s_prev*s_prev2*coeff)

def goertzel_power(samples, freq, fs):
    n = len(samples)
    k = int(0.5 + (n * freq) / fs)
    w = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(w)

    q0 = q1 = q2 = 0.0
    for s in samples:
        q0 = coeff * q1 - q2 + s
        q2 = q1
        q1 = q0

    return (q1*q1 + q2*q2 - q1*q2*coeff) / n

def confirm_dual_tone(samples, f1, f2):
    p1 = goertzel_power(samples, f1, SAMPLE_RATE)
    p2 = goertzel_power(samples, f2, SAMPLE_RATE)
    noise = np.mean(samples**2) + 1e-12

    if p1 < GOERTZEL_MIN_POWER or p2 < GOERTZEL_MIN_POWER:
        return False, p1, p2, noise

    if min(p1, p2) < noise * GOERTZEL_RATIO:
        return False, p1, p2, noise

    if (p1 + p2) < PAIR_POWER_SUM:
        return False, p1, p2, noise

    return True, p1, p2, noise

def spectral_flatness(mags):
    mags = np.array(mags) + 1e-12
    return np.exp(np.mean(np.log(mags))) / np.mean(mags)

# ================= SELCAL PAIR =================

def decode_selcal(samples, letters, freqs):
    mags = [goertzel_mag(samples, f, SAMPLE_RATE) for f in freqs]

    ranked = sorted(zip(letters, mags), key=lambda x: x[1], reverse=True)
    (l1, m1), (l2, m2) = ranked[:2]

    # third tone protection
    if ranked[2][1] > m2 * 0.85:
        dbg("REJECT third tone rivals second")
        return None

    if l1 == l2:
        return None

    i1 = letters.index(l1)
    i2 = letters.index(l2)

    # 🔒 CANONICAL ORDER (frequency-based, not magnitude)
    if i1 < i2:
        pair = l1 + l2
        f1, f2 = freqs[i1], freqs[i2]
    else:
        pair = l2 + l1
        f1, f2 = freqs[i2], freqs[i1]

    ok, p1, p2, noise = confirm_dual_tone(samples, f1, f2)
    dbg(f"CANDIDATE {pair} p1={p1:.2f} p2={p2:.2f}")

    if not ok:
        return None

    if (p1 + p2) < PAIR_POWER_MIN:
        dbg("REJECT low pair power")
        return None

    ratio = max(p1, p2) / max(min(p1, p2), 1e-6)
    if ratio > PAIR_IMBALANCE_MAX:
        dbg(f"REJECT imbalanced tones ratio={ratio:.2f}")
        return None

    flat = spectral_flatness(mags)
    if flat > FLATNESS_MAX:
        return None

    dbg(f"ACCEPT PAIR {pair}")
    return pair


# ================= LIVE LISTEN =================

def listen_live(letters, freqs, selcal_dict):
    pair_tracker = PairTracker()
    window_len = int(SAMPLE_RATE * WINDOW_SECONDS)
    buffer = np.zeros(window_len)
    write_pos = 0

    selcal_pairs = []
    last_full_code = None
    last_full_time = 0

    def callback(indata, frames, time_info, status):
        nonlocal write_pos
        mono = indata[:, 0]
        for s in mono:
            buffer[write_pos % window_len] = s
            write_pos += 1

    print("\nListening for ICAO SELCAL Codes... Ctrl-C to stop")

    with sd.InputStream(
        device=INPUT_DEVICE,
        channels=1,
        samplerate=SAMPLE_RATE,
        callback=callback
    ):
        try:
            while True:
                while True:
                    time.sleep(STEP_SECONDS)

                    samples = np.roll(buffer, -write_pos % window_len)
                    now = time.time()

                    # ---------- silence gate ----------
                    rms = np.sqrt(np.mean(samples**2))
                    if rms < SILENCE_RMS_MAX:
                        continue

                    # ---------- decode a pair ----------
                    pair = decode_selcal(samples, letters, freqs)
                    if pair is None:
                        continue

                    # ---------- join into SELCAL ----------
                    code = pair_tracker.update(pair, now)
                    if code is None:
                        continue

                    # ---------- validate ----------
                    if not is_valid_selcal(code):
                        continue

                    # ---------- lockout ----------
                    if (
                        code != last_full_code
                        or (now - last_full_time) >= FULL_CODE_LOCKOUT
                    ):
                        ts = datetime.now().strftime("%d/%m/%y %H:%M:%S")
                        desc = selcal_dict.get(code, "")
                        line = f"{ts} {code} {desc}"

                        print(line)
                        log_selcal(line)

                        last_full_code = code
                        last_full_time = now



        except KeyboardInterrupt:
            print("\nStopped.")



# ================= MAIN =================

if __name__ == "__main__":
    with open(JSON_FILE, "r") as f:
        selcal_json = json.load(f)

    selcal = selcal_json["SELCAL16"]
    letters = list(selcal.keys())
    freqs = list(selcal.values())

    selcal_dict = load_selcal_dictionary(DICT_FILE)
    listen_live(letters, freqs, selcal_dict)
