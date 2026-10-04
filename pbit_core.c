/* p-Bit-Netz fuer QUBO: E = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j ;  Update: x_i = 1 wenn tanh(-beta*dE_i/2) > r, r in (-1,1)
 * Q: dichte symmetrische Matrix (Diagonale = lineare Terme, Off-Diagonale = Kopplung je Paar, beide Haelften gefuellt).
 * Rauschen: r[(rep*(S+1) + s)*nfree + k]; Zeile 0 = Startzustand. Sequentielle Updates in fester Reihenfolge (-> Gibbs/Boltzmann).
 */
#include <math.h>
#include <stdlib.h>

static double energy(int n, const double *Q, const signed char *x) {
    double E = 0;
    for (int i = 0; i < n; i++) if (x[i]) {
        E += Q[i * n + i];
        for (int j = i + 1; j < n; j++) if (x[j]) E += Q[i * n + j];
    }
    return E;
}

/* Rueckgabe: solved_sweep[rep] = erster Sweep (1..S) mit E<0.5, sonst -1. xout[rep*n+i] = Zustand bei Loesung bzw. am Ende. Eend[rep] Endenergie. */
void pbit_run(int n, const double *Q, const signed char *clamp_mask, const signed char *clamp_val,
              int R, int S, const double *betas, const double *r, int mode,
              int *solved_sweep, signed char *xout, double *Eend) {
    int nfree = 0; int *freeidx = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) if (!clamp_mask[i]) freeidx[nfree++] = i;
    signed char *x = (signed char *)malloc(n);
    for (int rep = 0; rep < R; rep++) {
        const double *rr = r + (size_t)rep * (S + 1) * nfree;
        for (int i = 0; i < n; i++) x[i] = clamp_mask[i] ? clamp_val[i] : 0;
        for (int k = 0; k < nfree; k++) x[freeidx[k]] = rr[k] > 0 ? 1 : 0;
        int solved = -1;
        for (int s = 0; s < S && solved < 0; s++) {
            double b = betas[s];
            const double *rs = rr + (size_t)(s + 1) * nfree;
            for (int k = 0; k < nfree; k++) {
                int i = freeidx[k];
                double dE = Q[i * n + i];
                for (int j = 0; j < n; j++) if (j != i && x[j]) dE += Q[i * n + j];
                double th = tanh(-0.5 * b * dE);
                x[i] = (th > rs[k]) ? 1 : 0;
            }
            if (energy(n, Q, x) < 0.5) solved = s + 1;
        }
        solved_sweep[rep] = solved; Eend[rep] = energy(n, Q, x);
        for (int i = 0; i < n; i++) xout[(size_t)rep * n + i] = x[i];
    }
    free(x); free(freeidx);
}
