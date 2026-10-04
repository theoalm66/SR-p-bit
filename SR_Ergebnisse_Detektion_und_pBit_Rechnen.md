# SR-Projekt: Ergebnisse Detektion/Kodierung und p-Bit-Rechnen (Stand 2026-10-03)

Rauschquelle in allen Läufen: ESP32 + Dioden-Rauschgenerator (`noise_02.txt`, 1,8 Mio. Samples). Gemischte Hardware = Kontrolle für zeitliche Struktur.

## Teil A: Stochastische Resonanz (`sr_detect.py`)
Schwaches Signal (A = 0,6 bei Schwelle 1, allein nie auslösend), Rauschstärke σ wird durchgefahren.
- Detektion (Schwellendetektor): SNR-Spitze ca. 30 dB bei σ ≈ 0,64–0,73; Rand klein: kein Signal, Rand groß: 21 dB. Simulation = exakte Theorie (Median 0,05 dB).
- Kodieren (unterschwellige Bits, OOK): Bitfehlerrate 0,497 ohne Rauschen, Minimum ca. 0,018 bei σ ≈ 0,55 (ca. 0,87 bit/Symbol), bei viel Rauschen wieder steigend.
- Bistabile Doppelmulde: SNR-Spitze ca. 15–16 dB bei D ≈ 0,125 (Kramers-Erwartung ca. 0,10).
- Hardware (gauss-isiert) und gemischte Hardware unterscheiden sich nicht von PRNG (max |z| 1,9 / 2,3). Gefärbtes Rauschen fällt klar auf (Selbsttest).
- Rohe, nicht gauss-isierte Hardware-Verteilung (Schiefe −0,167, Exzess-Kurtosis −0,162) verschiebt die Kurve (−1,2 dB im Mittel, Spitze 30,5 dB). Die Theorie mit der eigenen empirischen Verteilung trifft sie im Median auf 0,04 dB (max. 0,23 dB bei Pegeln mit >= 50 Ereignissen/Lauf). Die Verteilungsform ist ein Stellrad der SR-Kurve; die zeitliche Struktur spielt keine Rolle.
- Bei der Kodierung ist die rohe Hardware etwas besser als PRNG (BER 0,0144 gegenüber 0,0187); gilt nur für diesen Aufbau.

## Teil B: p-Bits fürs Rechnen – Faktorisieren (`pbit_factor.py`)
Multiplizierer als QUBO (UND + Addierer), Produkt geklemmt, Faktorbits/innere Bits = p-Bits, Lösung = Energie 0, immer per Multiplikation geprüft.
- Skalierung (PRNG, 300 Läufe, 300 Sweeps): Erfolg 99 % (3 Bit), 35–52 % (4 Bit), 8–23 % (5 Bit), 0–2 % (6 Bit).
- Rauschpegel (N = 143): Erfolg bei festem β mit Spitze bei β = 2 (74 %); bei β <= 0,5 und bei β >= 10 nahe 0. Festes β = 2 schlägt Annealing 0,3 -> 3 (56 %).
- Quellen (N = 143, 150 Läufe, 200 Sweeps): PRNG 0,400; Hardware 0,353 (z −0,8); Hardware gemischt 0,453 (z +0,9); ohne Rauschen 0,000 (z −10). Unterschiede zwischen den Rauschquellen liegen im statistischen Rauschen (bei diesem Stichprobenumfang sind Unterschiede unter ca. 0,1 nicht auflösbar).
- Kein Geschwindigkeitsvorteil gegenüber Probedivision; gezeigt ist der Mechanismus (Rauschen als Rechenressource).

## Offen
- Größerer Quellenvergleich beim Faktorisieren (mehr Läufe, bessere Zeitpläne), p-Bit-Hardware-Parallelität.
- Weitere Instanzen/Optimierungsprobleme (MAX-CUT größer), Annealing-Zeitplan.
- T-CFSK-Bezug: Kettenbruch-Kicks brachten in der ILS keinen Vorteil (siehe `TCFSK_Kick_Experiment_Auswertung.md`).
