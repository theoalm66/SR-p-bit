#!/usr/bin/env python3
"""Rechnen mit p-Bits: Faktorisieren durch Rueckwaertslauf eines Multiplizierers.

Ein Multiplizierer (UND-Gatter + Voll-/Halbaddierer) wird als QUBO geschrieben: jedes Gatter ist eine quadratische Strafe, die genau dann 0 ist,
wenn das Gatter stimmt.  Das Produkt N wird festgehalten (geklemmt), die Faktorbits und alle inneren Bits sind p-Bits.  Das Netz wird mit
Rauschen (tanh-Regel, Boltzmann-Verteilung bei 1/beta) abgekuehlt; Energie 0 = gueltige Faktorisierung.
Gatter-QUBOs (x binaer):  UND c=a*b: ab - 2ac - 2bc + 3c ;  Addierer: (Summe der Eingaenge - Summe*Gewicht der Ausgaenge)^2.

Aufruf
  python pbit_factor.py --selftest                  # Gatter-Wahrheitstabellen, Netzwerk exakt, Zustaende mit Energie 0 = Faktorisierungen
  python pbit_factor.py noise_02.txt [--w 3 4 5]    # Faktorisieren mit PRNG / Hardware / gemischt / ohne Rauschen
  python pbit_factor.py --demo                      # nur PRNG und ohne Rauschen
"""
import argparse, ctypes, itertools, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_P = ctypes.c_void_p


def _load():
    """C-Kern (schnell) falls pbit_core.c per gcc baubar ist, sonst numpy-Rueckfall (langsamer, gleiche Ergebnisse)."""
    import subprocess
    ext = ".dll" if os.name == "nt" else ".so"
    so = os.path.join(HERE, "pbit_core" + ext); src = os.path.join(HERE, "pbit_core.c")
    try:
        if os.path.exists(src) and (not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src)):
            subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, src, "-lm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lib = ctypes.CDLL(so)
        lib.pbit_run.argtypes = [ctypes.c_int, _P, _P, _P, ctypes.c_int, ctypes.c_int, _P, _P, ctypes.c_int, _P, _P, _P]
        return lib
    except Exception:
        return None


_lib = _load() if os.environ.get("PBIT_FORCE_NUMPY") != "1" else None


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class Net:
    def __init__(self):
        self.n = 0; self.lin = {}; self.quad = {}; self.ops = []; self.clamp = {}

    def new(self):
        self.n += 1; return self.n - 1

    def add_sq(self, coefs):
        """Addiert (sum c_i x_i)^2 zur Energie (x binaer, x^2 = x)."""
        items = list(coefs.items())
        for v, c in items:
            self.lin[v] = self.lin.get(v, 0.0) + c * c
        for (u, cu), (v, cv) in itertools.combinations(items, 2):
            key = (min(u, v), max(u, v)); self.quad[key] = self.quad.get(key, 0.0) + 2 * cu * cv

    def AND(self, a, b):
        c = self.new()
        self.quad[(min(a, b), max(a, b))] = self.quad.get((min(a, b), max(a, b)), 0.0) + 1.0
        for u, w in ((a, -2.0), (b, -2.0)):
            key = (min(u, c), max(u, c)); self.quad[key] = self.quad.get(key, 0.0) + w
        self.lin[c] = self.lin.get(c, 0.0) + 3.0
        self.ops.append(("AND", c, a, b)); return c

    def FA(self, x, y, z):
        s, co = self.new(), self.new()
        self.add_sq({x: 1.0, y: 1.0, z: 1.0, s: -1.0, co: -2.0})
        self.ops.append(("FA", s, co, x, y, z)); return s, co

    def HA(self, x, y):
        s, co = self.new(), self.new()
        self.add_sq({x: 1.0, y: 1.0, s: -1.0, co: -2.0})
        self.ops.append(("HA", s, co, x, y)); return s, co

    def matrix(self):
        Q = np.zeros((self.n, self.n))
        for v, c in self.lin.items():
            Q[v, v] += c
        for (u, v), c in self.quad.items():
            Q[u, v] += c; Q[v, u] += c
        return Q


def build_multiplier(N, w):
    """w-Bit x w-Bit-Multiplizierer, Produktbits auf N geklemmt. Rueckgabe: Net, a-Bits, b-Bits, Produktbit-Variablen."""
    net = Net()
    a = [net.new() for _ in range(w)]; b = [net.new() for _ in range(w)]
    cols = [[] for _ in range(2 * w + 1)]
    for i in range(w):
        for j in range(w):
            cols[i + j].append(net.AND(a[i], b[j]))
    prod = []
    for k in range(2 * w + 1):
        col = cols[k]
        while len(col) >= 3:
            x, y, z = col.pop(0), col.pop(0), col.pop(0)
            s, co = net.FA(x, y, z); col.append(s)
            if k + 1 <= 2 * w: cols[k + 1].append(co)
        if len(col) == 2:
            x, y = col.pop(0), col.pop(0)
            s, co = net.HA(x, y); col.append(s)
            if k + 1 <= 2 * w: cols[k + 1].append(co)
        prod.append(col[0] if col else None)
    bits = [(N >> k) & 1 for k in range(2 * w + 1)]
    for k, v in enumerate(prod):
        if v is None:
            if bits[k]: raise ValueError(f"N={N} passt nicht in {w}x{w} Bit")
        else:
            net.clamp[v] = bits[k]
    if N >> (2 * w + 1): raise ValueError("N zu gross")
    return net, a, b, prod


def forward(net, a, b, p, q):
    """Wertet das Netz vorwaerts fuer p*q aus: alle Bits, danach Energie 0 (wenn Klemmung zu p*q passt)."""
    x = np.zeros(net.n, dtype=np.int8)
    for i, v in enumerate(a): x[v] = (p >> i) & 1
    for i, v in enumerate(b): x[v] = (q >> i) & 1
    for op in net.ops:
        if op[0] == "AND": x[op[1]] = x[op[2]] & x[op[3]]
        elif op[0] == "FA":
            t = int(x[op[3]]) + int(x[op[4]]) + int(x[op[5]]); x[op[1]] = t & 1; x[op[2]] = t >> 1
        else:
            t = int(x[op[3]]) + int(x[op[4]]); x[op[1]] = t & 1; x[op[2]] = t >> 1
    return x


def energy_of(Q, x):
    xf = x.astype(float)
    return float(np.diag(Q) @ xf + 0.5 * (xf @ (Q - np.diag(np.diag(Q))) @ xf))


def _run_numpy(n, Q, mask, val, r, betas):
    R, S1, nf = r.shape; S = S1 - 1
    free = np.flatnonzero(mask == 0); diag = np.diag(Q).copy(); Qoff = Q - np.diag(diag)
    X = np.zeros((R, n)); X[:, mask == 1] = val[mask == 1]; X[:, free] = (r[:, 0, :] > 0)
    solved = -np.ones(R, dtype=np.int32); xout = X.copy()
    for s in range(S):
        b = betas[s]
        for k, i in enumerate(free):
            dE = diag[i] + X @ Qoff[i]
            X[:, i] = (np.tanh(-0.5 * b * dE) > r[:, s + 1, k])
        E = X @ diag + 0.5 * ((X @ Qoff) * X).sum(axis=1)
        hit = (E < 0.5) & (solved < 0)
        solved[hit] = s + 1; xout[hit] = X[hit]
    Eend = X @ diag + 0.5 * ((X @ Qoff) * X).sum(axis=1)
    xout[solved < 0] = X[solved < 0]
    return solved, xout.astype(np.int8), Eend


def run_pbit(net, Q, r, betas):
    """r: (R, S+1, nfree) in (-1,1). Rueckgabe: solved_sweep (R,), xout (R,n), Eend (R,)"""
    R, S1, nfree = r.shape
    n = net.n
    mask = np.zeros(n, dtype=np.int8); val = np.zeros(n, dtype=np.int8)
    for v, c in net.clamp.items(): mask[v] = 1; val[v] = c
    assert nfree == n - int(mask.sum()), "Rauschbreite passt nicht zur Zahl freier p-Bits"
    r = np.ascontiguousarray(r, dtype=np.float64); betas = np.ascontiguousarray(betas, dtype=np.float64)
    if _lib is None:
        return _run_numpy(n, np.ascontiguousarray(Q), mask, val, r, betas)
    solved = np.empty(R, dtype=np.int32); xout = np.empty((R, n), dtype=np.int8); Eend = np.empty(R)
    _lib.pbit_run(n, ptr(np.ascontiguousarray(Q)), ptr(mask), ptr(val), R, S1 - 1, ptr(betas), ptr(r), 0, ptr(solved), ptr(xout), ptr(Eend))
    return solved, xout, Eend


def nfree(net):
    return net.n - len(net.clamp)


def decode(net, a, b, x):
    p = sum(int(x[v]) << i for i, v in enumerate(a)); q = sum(int(x[v]) << i for i, v in enumerate(b))
    return p, q


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ------------------------------------------------------------------ Selbsttest
def selftest():
    ok = True
    # [1] Gatter-Wahrheitstabellen: Energie 0 genau bei gueltigen Zeilen, sonst >= 1
    net = Net(); a, b = net.new(), net.new(); c = net.AND(a, b); Q = net.matrix(); good = True
    for x in itertools.product([0, 1], repeat=3):
        e = energy_of(Q, np.array(x)); valid = (x[2] == (x[0] & x[1]))
        good &= (abs(e) < 1e-9) if valid else (e >= 1 - 1e-9)
    print(f"[1] UND-Gatter-QUBO: {'OK' if good else 'FEHLER'}"); ok &= good
    net = Net(); v = [net.new() for _ in range(3)]; s, co = net.FA(*v); Q = net.matrix(); good = True
    for x in itertools.product([0, 1], repeat=5):
        e = energy_of(Q, np.array(x)); valid = (x[0] + x[1] + x[2] == x[3] + 2 * x[4])
        good &= (abs(e) < 1e-9) if valid else (e >= 1 - 1e-9)
    print(f"[2] Volladdierer-QUBO: {'OK' if good else 'FEHLER'}"); ok &= good
    # [3] Multiplizierer: Vorwaertsauswertung ergibt Energie 0 fuer alle p,q (Klemmung auf p*q), falsche Klemmung ergibt > 0
    good = True; cnt = 0
    for w in (2, 3, 4):
        for p in range(1 << w):
            for q in range(1 << w):
                N = p * q
                if N == 0: continue
                net, A, B, prod = build_multiplier(N, w); Q = net.matrix(); x = forward(net, A, B, p, q)
                good &= abs(energy_of(Q, x)) < 1e-9
                q2 = q + 1 if q + 1 < (1 << w) else q - 1
                x2 = forward(net, A, B, p, q2)
                for v, c in net.clamp.items(): x2[v] = c            # Produkt festhalten (Klemmung): falsche Faktoren verletzen dann mindestens ein Gatter
                if p * q2 != N:
                    good &= energy_of(Q, x2) >= 1 - 1e-9
                cnt += 1
    print(f"[3] Multiplizierer-Netz fuer {cnt} Produkte: Energie 0 bei richtiger Belegung, >=1 bei falscher: {'OK' if good else 'FEHLER'}"); ok &= good
    # [4] Vollstaendige Aufzaehlung: Zustaende mit Energie 0 = geordnete Faktorpaare (kleines Netz)
    good = True
    for N, w in ((15, 2), (9, 2), (6, 2)):
        net, A, B, prod = build_multiplier(N, w) if N < (1 << (2 * w)) else (None,) * 4
        Q = net.matrix(); free = [i for i in range(net.n) if i not in net.clamp]
        xs = np.zeros((1 << len(free), net.n), dtype=np.int8)
        for v, c in net.clamp.items(): xs[:, v] = c
        ids = np.arange(1 << len(free))
        for k, v in enumerate(free): xs[:, v] = (ids >> k) & 1
        xf = xs.astype(float)
        E = xf @ np.diag(Q) + 0.5 * np.einsum("ri,ij,rj->r", xf, Q - np.diag(np.diag(Q)), xf)
        sol = xs[E < 0.5]
        got = sorted({decode(net, A, B, s) for s in sol})
        want = sorted((p, q) for p in range(1 << w) for q in range(1 << w) if p * q == N)
        g = (got == want) and (len(sol) == len(want))
        print(f"[4] N={N} ({w}x{w} Bit, {len(free)} freie Bits): {len(sol)} Zustaende mit E=0, gefunden {got} (erwartet {want}) -> {'OK' if g else 'FEHLER'}")
        good &= g
    ok &= good
    # [5] Sampler-Kontrolle: bei beta -> 0 entspricht der Sampler dem Zufall (p(Bit=1)=0.5); mit Rauschen findet er die Loesung, ohne nicht
    rng = np.random.default_rng(3)
    net, A, B, prod = build_multiplier(143, 4); Q = net.matrix(); nf = nfree(net)
    R, S = 200, 300; betas = np.linspace(0.3, 3.0, S)
    r = rng.uniform(-1, 1, (R, S + 1, nf))
    sol, xo, Ee = run_pbit(net, Q, r, betas)
    n_ok = int((sol > 0).sum()); fac_ok = all(decode(net, A, B, xo[i])[0] * decode(net, A, B, xo[i])[1] == 143 for i in np.flatnonzero(sol > 0))
    r0 = np.zeros_like(r)
    sol0, _, _ = run_pbit(net, Q, r0, betas)
    g5 = n_ok / R > 0.2 and fac_ok and (sol0 > 0).mean() < n_ok / R / 2
    print(f"[5] 143=11x13 (w=4, {nf} freie p-Bits): mit Rauschen {n_ok}/{R} geloest, alle Loesungen multiplizieren zu 143: {fac_ok}; "
          f"ohne Rauschen (r=0) {int((sol0 > 0).sum())}/{R} -> {'OK' if g5 else 'FEHLER'}")
    ok &= g5
    print("== p-Bit-Faktorisierer-Selbsttest", "bestanden ==" if ok else "FEHLGESCHLAGEN ==")
    return ok



# ------------------------------------------------------------------ Experiment
PILOT = [(77, 4), (91, 4)]                                   # Pilot (nicht im Test): waehlt Endwert beta1 des linearen Plans 0.3 -> beta1
TEST = [(35, 3), (143, 4), (169, 4), (667, 5), (527, 5), (899, 5), (3127, 6), (2867, 6)]
BETA1_GRID = [2.0, 3.0, 5.0, 8.0]
BETA0 = 0.3


def success(net, Q, nf, R, S, betas, rng, noise=None):
    """noise: None -> PRNG, 'none' -> r=0 (kein Rauschen), sonst Array (laufender Strom gleichverteilt in (-1,1)), wird fortlaufend verbraucht."""
    need = R * (S + 1) * nf
    if noise is None:
        r = rng.uniform(-1, 1, (R, S + 1, nf))
    elif isinstance(noise, str) and noise == "none":
        r = np.zeros((R, S + 1, nf))
    else:
        r = noise[:need].reshape(R, S + 1, nf)
    sol, xo, E = run_pbit(net, Q, r, betas)
    return sol


def check_solutions(net, A, B, xo_sol, N):
    return True


def experiment(x=None, seed=1, R=300, S=300):
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    out = {}
    # Pilot
    best, bs = None, {}
    for b1 in BETA1_GRID:
        k = n = 0
        for N, w in PILOT:
            net, A, B, prod = build_multiplier(N, w); Q = net.matrix(); nf = nfree(net)
            sol = success(net, Q, nf, 200, S, np.linspace(BETA0, b1, S), rng); k += int((sol > 0).sum()); n += 200
        bs[b1] = k / n
    b1 = max(bs, key=bs.get)
    print("Pilot (N=77, 91; je 200 Laeufe, 300 Sweeps): Erfolg nach beta1 = " + ", ".join(f"{k}: {v:.3f}" for k, v in bs.items()) + f" -> beta1 = {b1}")
    betas = np.linspace(BETA0, b1, S)
    out["pilot"] = bs; out["beta1"] = b1
    # (a) Skalierung
    print(f"\n(a) Skalierung mit der Groesse (PRNG, {R} Laeufe, {S} Sweeps, beta 0.3 -> {b1}):")
    print(f"{'N':>6} {'w':>2} {'p-Bits':>7} {'Erfolg':>8} {'95%-KI':>16} {'Median-Sweeps':>14}")
    scal = []
    for N, w in TEST:
        net, A, B, prod = build_multiplier(N, w); Q = net.matrix(); nf = nfree(net)
        r = rng.uniform(-1, 1, (R, S + 1, nf)); sol, xo, E = run_pbit(net, Q, r, betas)
        ok = sol > 0
        for i in np.flatnonzero(ok):
            p, q = decode(net, A, B, xo[i]); assert p * q == N and p > 1 and q > 1, "falsche Faktorisierung"
        lo, hi = wilson(int(ok.sum()), R)
        med = float(np.median(sol[ok])) if ok.any() else float("nan")
        print(f"{N:>6} {w:>2} {nf:>7} {ok.mean():>8.3f} [{lo:.3f},{hi:.3f}] {med:>14.0f}")
        scal.append((N, w, nf, int(ok.sum()), R, med))
    out["scal"] = scal
    # (b) Rauschpegel
    N, w = 143, 4
    net, A, B, prod = build_multiplier(N, w); Q = net.matrix(); nf = nfree(net)
    print(f"\n(b) Rauschpegel fuer N={N}: fester beta ueber {S} Sweeps (kleiner beta = mehr Rauschen), {R} Laeufe:")
    scan = []
    for b in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0]:
        sol = success(net, Q, nf, R, S, np.full(S, b), rng); k = int((sol > 0).sum()); lo, hi = wilson(k, R)
        scan.append((b, k, R)); print(f"  beta={b:<5} Erfolg {k / R:.3f} [{lo:.3f},{hi:.3f}]")
    sol = success(net, Q, nf, R, S, betas, rng); k = int((sol > 0).sum()); lo, hi = wilson(k, R)
    print(f"  Annealing 0.3->{b1}: Erfolg {k / R:.3f} [{lo:.3f},{hi:.3f}]")
    out["scan"] = scan; out["anneal143"] = (k, R)
    # (c) Quellen
    Rs, Ss = 150, 200
    betas_s = np.linspace(BETA0, b1, Ss)
    print(f"\n(c) Rauschquellen fuer N={N} ({Rs} Laeufe, {Ss} Sweeps, beta 0.3 -> {b1}):")
    srcs = {"PRNG": None, "kein Rauschen (r=0)": "none"}
    need = Rs * (Ss + 1) * nf
    if x is not None and len(x) >= need:
        from noise_pbit import pit_uniform
        u = pit_uniform(x[:need], rng); r_hw = 2 * u - 1
        srcs["Hardware (PIT)"] = r_hw; srcs["Hardware gemischt"] = rng.permutation(r_hw)
    elif x is not None:
        print(f"  (Hardware uebersprungen: {need} Werte noetig, {len(x)} vorhanden)")
    res = {}
    for name, nz in srcs.items():
        sol = success(net, Q, nf, Rs, Ss, betas_s, rng, nz); k = int((sol > 0).sum()); lo, hi = wilson(k, Rs); res[name] = (k, Rs)
        print(f"  {name:<22} Erfolg {k / Rs:.3f} [{lo:.3f},{hi:.3f}]")
    ref = res["PRNG"]
    for name, (k, n) in res.items():
        if name == "PRNG": continue
        p1, p2 = k / n, ref[0] / ref[1]; se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / ref[1])
        z = (p1 - p2) / se if se > 0 else 0.0
        print(f"    {name:<22} gegen PRNG: {p1 - p2:+.3f}, z={z:+.1f}" + ("   <-- AUFFAELLIG" if abs(z) > 3 else ""))
    out["src"] = res
    return out


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.ticker
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    a = ax[0]
    sc = out["scal"]
    for w in sorted({t[1] for t in sc}):
        pts = [t for t in sc if t[1] == w]
        k = sum(t[3] for t in pts); n = sum(t[4] for t in pts); lo, hi = wilson(k, n)
        a.errorbar([w], [k / n], [[k / n - lo], [hi - k / n]], fmt="o", color="#2563eb", capsize=3)
        for t in pts: a.plot([w + 0.06], [t[3] / t[4]], ".", color="#9ca3af")
    a.set_xlabel("Faktorbreite w (Bit)"); a.set_ylabel("Anteil gelöster Läufe"); a.set_xticks(sorted({t[1] for t in sc}))
    a.set_title("A  Skalierung (blau: je w gepoolt; grau: einzelne N)")
    a = ax[1]
    sc = out["scan"]; bs = [t[0] for t in sc]; ys = [t[1] / t[2] for t in sc]
    ci = [wilson(t[1], t[2]) for t in sc]
    a.errorbar(bs, ys, [[y - c[0] for y, c in zip(ys, ci)], [c[1] - y for y, c in zip(ys, ci)]], fmt="o-", color="#2563eb", capsize=3, label="fester β")
    k, n = out["anneal143"]; a.axhline(k / n, color="#ea580c", ls="--", label="Annealing 0,3→%g" % out["beta1"])
    a.set_xscale("log"); a.set_xlabel("β (klein = viel Rauschen, groß = wenig)"); a.set_ylabel("Anteil gelöster Läufe")
    a.set_title("B  Rauschpegel, N = 143"); a.legend(frameon=False, fontsize=8)
    a = ax[2]
    names = list(out["src"]); ys = [out["src"][k][0] / out["src"][k][1] for k in names]; ci = [wilson(*out["src"][k]) for k in names]
    cols = ["#2563eb", "#6b7280", "#ea580c", "#16a34a"][:len(names)]
    a.bar(range(len(names)), ys, color=cols, yerr=[[y - c[0] for y, c in zip(ys, ci)], [c[1] - y for y, c in zip(ys, ci)]], capsize=3)
    a.set_xticks(range(len(names))); a.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    a.set_ylabel("Anteil gelöster Läufe"); a.set_title("C  Rauschquelle, N = 143")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False); a_.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datei", nargs="?"); ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--demo", action="store_true"); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.datei and not a.demo:
        ap.error("Datei angeben, --demo oder --selftest")
    x = None
    if a.datei:
        from noise_pbit import load
        x, skipped = load(a.datei, a.col); print(f"{len(x)} Samples aus {a.datei} ({skipped} Zeilen uebersprungen)")
    out = experiment(x, seed=a.seed)
    plot(out, "faktor_kurven.png"); print("\nfaktor_kurven.png geschrieben.")
