#!/usr/bin/env python3
"""p-bit Ising-Maschine fuer MAX-CUT (Richtung 2) - mit Selbsttest.

p-bit:   m_i = sgn( tanh(beta * I_i) - r ),  r ~ U(-1,1)   (Camsari et al.)
Eingang: I_i = -sum_j W_ij m_j   (Max-Cut: Energie E = sum_edges w_ij m_i m_j, minimieren)
Updates: sequentiell (Gibbs-aequivalent) -> stationaer: Boltzmann P ~ exp(-beta E).
Hardware-Bild: tanh(beta*I) - r  == Komparator, der I mit Rauschen r vergleicht.
"""
import itertools
import numpy as np

# 4-Knoten-Graph: Ring 0-1-2-3-0 plus Diagonale 0-2 (gewichtet, Max-Cut = 4 bei w=1: zwei Dreiecke, je eine Kante bleibt ungeschnitten)
EDGES = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 1.0)]
N = 4


def weights(edges=EDGES, n=N):
    W = np.zeros((n, n))
    for i, j, w in edges:
        W[i, j] = W[j, i] = w
    return W


def energy(m, W):
    return 0.5 * m @ W @ m          # = sum_edges w m_i m_j


def cut_value(m, W):
    return 0.25 * np.sum(W * (1 - np.outer(m, m)))


def sweep(m, W, beta, rng):
    for i in rng.permutation(len(m)):
        I = -W[i] @ m                # lokales Feld
        m[i] = 1.0 if np.tanh(beta * I) > rng.uniform(-1, 1) else -1.0
    return m


def sample(W, beta, n_sweeps, rng, burn=200):
    m = rng.choice([-1.0, 1.0], size=len(W))
    counts = {}
    for t in range(n_sweeps + burn):
        m = sweep(m, W, beta, rng)
        if t >= burn:
            key = tuple(int(x) for x in m)
            counts[key] = counts.get(key, 0) + 1
    return counts


def exact_boltzmann(W, beta):
    states = [np.array(s, float) for s in itertools.product([-1, 1], repeat=len(W))]
    e = np.array([energy(s, W) for s in states])
    p = np.exp(-beta * (e - e.min()))
    p /= p.sum()
    return {tuple(int(x) for x in s): pi for s, pi in zip(states, p)}


def tv_distance(counts, exact):
    tot = sum(counts.values())
    return 0.5 * sum(abs(counts.get(k, 0) / tot - p) for k, p in exact.items())


def anneal(W, rng, sweeps=60, b0=0.1, b1=3.0):
    m = rng.choice([-1.0, 1.0], size=len(W))
    for beta in np.linspace(b0, b1, sweeps):
        m = sweep(m, W, beta, rng)
    return m


def success_rate(W, rng, trials=2000, **kw):
    states = [np.array(s, float) for s in itertools.product([-1, 1], repeat=len(W))]
    best = max(cut_value(s, W) for s in states)
    hit = sum(cut_value(anneal(W, rng, **kw), W) >= best - 1e-9 for _ in range(trials))
    n_opt = sum(cut_value(s, W) >= best - 1e-9 for s in states)
    return hit / trials, n_opt / len(states), best


if __name__ == "__main__":
    rng = np.random.default_rng(12345)       # praeregistriert
    W = weights()

    print("== SELBSTTEST ==")
    # Positivkontrolle: Statistik == exakte Boltzmann-Verteilung (beta = 0.7)
    tv = tv_distance(sample(W, 0.7, 40000, rng), exact_boltzmann(W, 0.7))
    print(f"[Positiv] TV(empirisch, Boltzmann) bei beta=0.7: {tv:.4f}  (Soll < 0.02)")
    assert tv < 0.02
    # Nullkontrolle: beta=0 -> Gleichverteilung (reines Rauschen)
    tv0 = tv_distance(sample(W, 0.0, 40000, rng), exact_boltzmann(W, 0.0))
    print(f"[Null]    TV(empirisch, uniform)   bei beta=0  : {tv0:.4f}  (Soll < 0.02)")
    assert tv0 < 0.02
    # Zufallsbaseline: Anteil optimaler Zustaende
    sr, base, best = success_rate(W, rng)
    print(f"[Annealing] Max-Cut={best:.0f}; Trefferquote {sr:.3f} vs. Zufall {base:.3f}")
    assert sr > 3 * base

    print("\n== beta-Scan (Trefferquote bei festem beta, 'Rauschpegel' = 1/beta) ==")
    for b in [0.05, 0.2, 0.5, 1.0, 2.0, 4.0]:
        hit = 0
        states = [np.array(s, float) for s in itertools.product([-1, 1], repeat=N)]
        for _ in range(1500):
            m = rng.choice([-1.0, 1.0], size=N)
            for _ in range(30):
                m = sweep(m, W, b, rng)
            hit += cut_value(m, W) >= best - 1e-9
        print(f"  beta={b:5.2f}  Trefferquote={hit/1500:.3f}")
