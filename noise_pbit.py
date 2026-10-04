#!/usr/bin/env python3
"""Rauschquelle -> p-Bit: Qualitaetspruefung und funktionaler Test.

Eingabe: Textdatei mit ADC-Samples (eine Zahl pro Zeile ODER mehrere Spalten, --col waehlt die Spalte;
Zeilen, die keine Zahl enthalten (Header, Kommentare), werden uebersprungen).

Ablauf
  1. Diagnose: Wertevorrat/Ties, Autokorrelation (Ljung-Box), Spektrum (Steigung, Linien),
     Stationaritaet (Mittel/Streuung in 4 Bloecken, Verteilung 1. vs. 2. Haelfte).
  2. Aufbereitung: randomisierte Wahrscheinlichkeits-Integraltransformation (PIT) -> r in (-1,1),
     optional Ausduennen (--fix thin) oder AR-Vorfilterung (--fix ar).
  3. Funktionstest: p-Bit-Netz (pbit_maxcut.py) bekommt r NUR aus dem Rauschstrom (feste Update-
     Reihenfolge, kein PRNG im Update) und wird mit der exakten Boltzmann-Verteilung verglichen.
     Referenz: dieselbe Pipeline mit PRNG-Strom gleicher Laenge (Verteilung der TV-Abstaende).

Aufruf
  python noise_pbit.py --selftest
  python noise_pbit.py messung.txt --col 1 [--fix thin|ar|none] [--beta 0.7] [--acf-tol 0.02]
"""
import argparse
import math
import numpy as np
from pbit_maxcut import weights, exact_boltzmann, tv_distance

# ----------------------------------------------------------------------------- Laden / Simulation

def load(path, col=0):
    vals, skipped = [], 0
    with open(path, errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%/":
                skipped += 1
                continue
            parts = s.replace(",", " ").replace(";", " ").split()
            try:
                vals.append(float(parts[col]))
            except (IndexError, ValueError):
                skipped += 1
    return np.array(vals), skipped


def simulate(kind, n, rng, sigma=150.0):
    if kind == "white":
        x = rng.normal(0, sigma, n)
    elif kind == "colored":                       # AR(1), phi=0.9: z.B. Tiefpass im Analogpfad
        e = rng.normal(0, 1, n)
        x = np.empty(n); x[0] = e[0]
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + e[t]
        x *= sigma / x.std()
    elif kind == "weak":                          # AR(1), phi=0.012: formal nicht weiss, praktisch egal
        e = rng.normal(0, 1, n)
        x = np.empty(n); x[0] = e[0]
        for t in range(1, n):
            x[t] = 0.012 * x[t - 1] + e[t]
        x *= sigma / x.std()
    elif kind == "drift":                         # weisses Rauschen + langsame Drift (Versorgung/Temperatur)
        t = np.arange(n)
        x = rng.normal(0, sigma, n) + 0.6 * sigma * np.sin(2 * np.pi * t / (n / 3.3))
    else:
        raise ValueError(kind)
    return np.round(x + 2048)                     # 12-Bit-Codes

# ----------------------------------------------------------------------------- Diagnose

def acf(x, nlag):
    x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[: nlag + 1]
    return a / a[0]


def ljung_box_p(x, h=20):
    n = len(x); r = acf(x, h)[1:]
    Q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, h + 1)))
    z = ((Q / h) ** (1 / 3) - (1 - 2 / (9 * h))) / math.sqrt(2 / (9 * h))   # Wilson-Hilferty
    return 0.5 * math.erfc(z / math.sqrt(2)), float(np.max(np.abs(r)))


def psd(x, seg=1024):
    x = x - x.mean(); w = np.hanning(seg); k = len(x) // seg
    P = np.mean([np.abs(np.fft.rfft(x[i * seg:(i + 1) * seg] * w)) ** 2 for i in range(k)], axis=0)
    return np.fft.rfftfreq(seg)[1:], P[1:]


def spectrum_report(x):
    f, P = psd(x)
    slope = np.polyfit(np.log(f), np.log(P), 1)[0]           # 0 = weiss, -1 = 1/f, -2 = Braun
    peak = P.max() / np.median(P)
    return slope, peak, f[P.argmax()]


def ks2(a, b):
    sa, sb = np.sort(a), np.sort(b); allv = np.concatenate([sa, sb])
    return float(np.max(np.abs(np.searchsorted(sa, allv, "right") / len(a)
                               - np.searchsorted(sb, allv, "right") / len(b))))


def diagnose(x, fs=None):
    n = len(x); out = {}
    vals, cnt = np.unique(x, return_counts=True)
    out["distinct"] = len(vals); out["maxshare"] = cnt.max() / n
    out["topcode"] = float(vals[cnt.argmax()])
    out["clip"] = float(np.mean((x <= 0) | (x >= 4095)))
    a40 = acf(x, 40)[1:]; idx = np.argsort(-np.abs(a40))[:5]
    out["toplags"] = [(int(i) + 1, float(a40[i])) for i in idx]
    out["acf_sigma"] = 1 / math.sqrt(n)
    out["mean"], out["std"] = x.mean(), x.std()
    out["lb_p"], out["acf_max"] = ljung_box_p(x)
    out["acf1"] = acf(x, 1)[1]
    out["slope"], out["peak"], out["peakf"] = spectrum_report(x)
    blocks = np.array_split(x, 4)
    bm = np.array([b.mean() for b in blocks]); bs = np.array([b.std() for b in blocks])
    out["drift_z"] = float(np.max(np.abs(bm - x.mean())) / (x.std() / math.sqrt(n / 4)))
    out["std_ratio"] = float(bs.max() / bs.min())
    D = ks2(x[: n // 2], x[n // 2:])
    out["ks_D"], out["ks_crit"] = D, 1.63 * math.sqrt(4 / n)
    return out


def whiteness(d, tol):
    """'weiss' (Test bestanden) | 'praktisch weiss' (Test faellt durch, Effekt < tol) | 'nicht weiss'"""
    if d["lb_p"] > 0.01:
        return "weiss"
    return "praktisch weiss" if d["acf_max"] < tol else "nicht weiss"


def print_diag(d, label, tol=0.02):
    level = whiteness(d, tol)
    ok_stat = d["drift_z"] < 4 and d["std_ratio"] < 1.15 and d["ks_D"] < d["ks_crit"]
    print(f"--- Diagnose: {label}")
    print(f"  Codes: {d['distinct']} verschiedene; haeufigster Code {d['topcode']:.0f} mit {d['maxshare']:.3%} (Gauss-Erwartung ~{0.4 / max(d['std'], 1):.3%});"
          f"  bei 0/4095: {d['clip']:.3%};  mean={d['mean']:.1f}  std={d['std']:.1f}")
    print(f"  Autokorrelation: lag1={d['acf1']:+.3f}, max|lag1..20|={d['acf_max']:.4f} (Zufallsstreuung/Lag ~{d['acf_sigma']:.4f}), Ljung-Box p={d['lb_p']:.2g}  -> {level.upper() if level != 'weiss' else 'weiss'}"
          + (f"  (Effekt unter Toleranz {tol})" if level == "praktisch weiss" else ""))
    print("  auffaelligste Lags (1..40): " + ", ".join(f"{l}:{v:+.4f}" for l, v in d["toplags"]))
    print(f"  Spektrum: Steigung={d['slope']:+.2f} (0=weiss, -1=1/f), staerkste Linie = {d['peak']:.1f}x Median bei f/fs={d['peakf']:.4f}")
    print(f"  Stationaritaet: Drift z={d['drift_z']:.1f}, Streuungsverhaeltnis={d['std_ratio']:.2f}, KS(1./2. Haelfte) D={d['ks_D']:.4f} (krit {d['ks_crit']:.4f})  -> {'stationaer' if ok_stat else 'NICHT stationaer'}")
    return level != "nicht weiss", ok_stat


# ----------------------------------------------------------------------------- Aufbereitung

def thin(x, tol=0.02, kmax=200, min_n=4000):
    for k in range(1, kmax + 1):
        y = x[::k]
        if len(y) < min_n:
            return None, None
        p, amax = ljung_box_p(y)
        if p > 0.01 or amax < tol:
            return y, k
    return None, None


def prewhiten(x, order=16):
    x = x - x.mean(); n = len(x)
    X = np.column_stack([x[order - i - 1: n - i - 1] for i in range(order)])
    a, *_ = np.linalg.lstsq(X, x[order:], rcond=None)
    return x[order:] - X @ a


def pit_uniform(x, rng):
    """Randomisierte PIT: Ties werden innerhalb ihres Code-Intervalls gleichverteilt aufgeloest.
    Hinweis: diese Dither-Zufallszahl kommt aus dem PRNG; sie wirkt nur INNERHALB eines ADC-Codes."""
    n = len(x); sx = np.sort(x)
    lo = np.searchsorted(sx, x, "left") / n; hi = np.searchsorted(sx, x, "right") / n
    return lo + rng.random(n) * (hi - lo)

# ----------------------------------------------------------------------------- p-Bit mit externem Rauschen

def run_pbit(W, beta, r, burn=200):
    n = len(W); m = np.ones(n); counts = {}; k = 0
    nsw = len(r) // n
    for t in range(nsw):
        for i in range(n):
            m[i] = 1.0 if math.tanh(beta * -(W[i] @ m)) > r[k] else -1.0
            k += 1
        if t >= burn:
            key = tuple(int(v) for v in m); counts[key] = counts.get(key, 0) + 1
    return counts, nsw - burn


def functional_test(W, beta, streams, ref_len, n_base, rng, exact):
    base = []
    for _ in range(n_base):
        c, _ = run_pbit(W, beta, rng.uniform(-1, 1, ref_len)); base.append(tv_distance(c, exact))
    base = np.array(base); lim = np.percentile(base, 95)
    print(f"  Referenz (PRNG, {n_base} Laeufe, {ref_len // len(W) - 200} Sweeps): TV = {base.mean():.4f} +- {base.std():.4f}, 95%-Grenze {lim:.4f}")
    res = {}
    for name, r in streams.items():
        c, ns = run_pbit(W, beta, r[:ref_len]); tv = tv_distance(c, exact)
        res[name] = tv
        print(f"  {name:<28s} TV = {tv:.4f}  -> {'OK' if tv <= lim else 'ABWEICHUNG'}")
    return res, lim

# ----------------------------------------------------------------------------- Hauptablauf

def evaluate(x, label, args, rng):
    d = diagnose(x)
    ok_white, ok_stat = print_diag(d, label, args.acf_tol)
    level = whiteness(d, args.acf_tol)
    u_raw = pit_uniform(x, rng)
    streams = {"A roh (PIT)": 2 * u_raw - 1}
    note = ""
    if args.fix == "thin" and level != "nicht weiss":
        note = f"Ausduennen nicht noetig ({level}, max|acf|={d['acf_max']:.4f} < {args.acf_tol}) - Rohstrom A wird verwendet"
    elif args.fix == "thin":
        y, k = thin(x, args.acf_tol)
        if y is None:
            note = "Ausduennen bis k=200 reicht nicht (Linien/Drift?) -> kein Strom B"
        else:
            note = f"Ausduennen mit k={k}: {len(y)} von {len(x)} Samples bleiben ({100 * len(y) / len(x):.1f} %)"
            streams["B ausgeduennt (PIT)"] = 2 * pit_uniform(y, rng) - 1
    elif args.fix == "ar":
        e = prewhiten(x, args.ar_order)
        p = ljung_box_p(e)[0]
        note = f"AR({args.ar_order})-Vorfilterung: Ljung-Box p danach = {p:.2g}"
        streams["B AR-vorgefiltert (PIT)"] = 2 * pit_uniform(e, rng) - 1
    streams["C gemischt (Kontrolle)"] = 2 * u_raw[rng.permutation(len(u_raw))] - 1
    if note:
        print("  " + note)
    W = weights(); exact = exact_boltzmann(W, args.beta)
    ref_len = min(len(s) for s in streams.values())
    print(f"--- Funktionstest bei beta={args.beta}: p-Bit-Netz nur mit Rauschstrom ({ref_len} Werte)")
    res, lim = functional_test(W, args.beta, streams, ref_len, args.baselines, rng, exact)
    return d, res, lim, ok_white, ok_stat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datei", nargs="?")
    ap.add_argument("--col", type=int, default=0, help="Spalte (0-basiert)")
    ap.add_argument("--fix", choices=["thin", "ar", "none"], default="thin")
    ap.add_argument("--ar-order", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--acf-tol", type=float, default=0.02,
                    help="Effektgroesse: max|Autokorrelation| darunter gilt als praktisch weiss (Standard 0.02)")
    ap.add_argument("--baselines", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=400000, help="Samples im Selbsttest")
    args = ap.parse_args()
    rng = np.random.default_rng(2026)

    if args.selftest:
        print("== SELBSTTEST mit simulierten Quellen (12-Bit-Codes, sigma=150) ==")
        out = {}
        for kind in ["white", "weak", "colored", "drift"]:
            print(f"\n######## Quelle: {kind}")
            out[kind] = evaluate(simulate(kind, args.n, rng), kind, args, rng)
        _, rw, lw, okw, oks = out["white"]
        dwk, rwk, lwk, okwk, _ = out["weak"]
        assert whiteness(dwk, args.acf_tol) == "praktisch weiss" and rwk["A roh (PIT)"] <= lwk, \
            "schwache Faerbung muss als praktisch weiss gelten und den Funktionstest bestehen"
        _, rc, lc, okcw, _ = out["colored"]
        _, rd, ld, _, okds = out["drift"]
        assert okw and oks and rw["A roh (PIT)"] <= lw, "weisse Quelle muss bestehen (Positivkontrolle)"
        assert not okcw, "gefaerbte Quelle muss als nicht weiss erkannt werden (Negativkontrolle)"
        assert not okds, "Drift muss als nicht stationaer erkannt werden"
        print("\n== Selbsttest bestanden: weiss -> OK, schwach gefaerbt -> praktisch weiss, stark gefaerbt -> erkannt, Drift -> erkannt ==")
        return

    if not args.datei:
        ap.error("Datei angeben oder --selftest")
    x, skipped = load(args.datei, args.col)
    print(f"{len(x)} Samples aus {args.datei} (Spalte {args.col}), {skipped} Zeilen uebersprungen")
    if len(x) < 20000:
        print("Warnung: <20000 Samples - Tests haben wenig Aussagekraft.")
    evaluate(x, args.datei, args, rng)


if __name__ == "__main__":
    main()
