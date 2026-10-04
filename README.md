# SR-p-bit

Stochastische Resonanz und p-Bit-Rechnen: Experimente dazu, wie Rauschen nicht nur ein Störfaktor ist, sondern als Rechenressource genutzt werden kann — von der Signaldetektion mit Rauschverstärkung bis zu einem probabilistischen "Computer" aus p-Bits, der Optimierungsprobleme und sogar Faktorisierung löst.

**[→ p-Bit-Labor öffnen](./p-Bit-Labor.html)** — interaktives Labor im Browser, keine Installation nötig.

## Inhalt

| Datei | Beschreibung |
|---|---|
| [`p-Bit-Labor.html`](./p-Bit-Labor.html) | Interaktives Browser-Labor: ein einzelnes p-Bit (Komparator + Rauschen) live beobachten, sechs p-Bits zu einem MAX-CUT-Netz koppeln, Annealing starten, Kontrolle gegen die exakte Boltzmann-Verteilung. |
| [`pbit_maxcut.py`](./pbit_maxcut.py) | p-Bit-Ising-Maschine für MAX-CUT mit Selbsttest (Vergleich gegen exakte Boltzmann-Verteilung, Null- und Positivkontrolle). |
| [`pbit_factor.py`](./pbit_factor.py) | Faktorisieren durch Rückwärtslauf eines Multiplizierers: Der Multiplizierer wird als QUBO kodiert (UND-Gatter + Addierer als quadratische Strafterme), das Produkt wird festgehalten, die Faktorbits sind p-Bits. Nutzt optional einen schnellen C-Kern (`pbit_core.c`). |
| [`pbit_core.c`](./pbit_core.c) | C-Kern für die p-Bit-Updates, wird von `pbit_factor.py` automatisch gebaut (gcc) und geladen, falls verfügbar; sonst numpy-Fallback mit identischen Ergebnissen. |
| [`noise_pbit.py`](./noise_pbit.py) | Qualitätsprüfung echter Rauschquellen (ADC-Samples) und Funktionstest: Taugt das Rauschen als p-Bit-Antrieb? Diagnose (Autokorrelation, Spektrum, Stationarität), PIT-Aufbereitung, Vergleich gegen PRNG-Referenz. |
| [`noise_02.txt`](./noise_02.txt) | Rohdaten einer Hardware-Rauschquelle (ESP32 + Dioden-Rauschgenerator, ca. 1,8 Mio. Samples) als Eingabe für `noise_pbit.py` / `pbit_factor.py`. |
| [`SR_Ergebnisse_Detektion_und_pBit_Rechnen.md`](./SR_Ergebnisse_Detektion_und_pBit_Rechnen.md) | Ergebniszusammenfassung: Detektion/Kodierung per stochastischer Resonanz und Faktorisieren mit p-Bits, inkl. Kennzahlen zu Erfolgsraten und Rauschquellenvergleich. |
| [`faktor_kurven.png`](./faktor_kurven.png) | Kurvenplot zu den Faktorisierungs-Experimenten. |
| [`SR-Projekte-detect-signal-p-Bit-Rechner.pdf`](./SR-Projekte-detect-signal-p-Bit-Rechner.pdf), [`SR_thermo_Projektkontext.pdf`](./SR_thermo_Projektkontext.pdf) | Projektunterlagen / thermodynamischer Hintergrund. |

## Die Grundidee

Ein **p-Bit** ("probabilistisches Bit") ist ein Komparator: Eingang plus Rauschen wird mit einer Schwelle verglichen. Das Ergebnis flackert zufällig zwischen 0 und 1, mit einer Wahrscheinlichkeit, die vom Eingang abhängt — eine weiche, verrauschte Sigmoid-Kennlinie statt eines harten Schalters.

Koppelt man mehrere p-Bits lokal (jeder Knoten bekommt als Eingang eine gewichtete Summe seiner Nachbarn), stellt sich eine Boltzmann-Verteilung über die Netzwerkzustände ein. Das reicht aus, um:

- **MAX-CUT** zu lösen (Annealing: Rauschen/Temperatur langsam absenken, das Netz "friert" in Zuständen mit hohem Cut ein),
- **stochastische Resonanz** zur Signaldetektion zu nutzen (ein unterschwelliges Signal wird durch die richtige Menge Rauschen überhaupt erst messbar),
- einen **Multiplizierer rückwärts laufen zu lassen** und so kleine Zahlen zu faktorisieren.

Wichtig: Ein p-Bit hat immer einen festen, nur zufällig wechselnden Wert — keine Superposition, keine Verschränkung, kein Quantenvorteil. Die Rechenzeit wächst klassisch mit der Problemgröße. Gezeigt wird der Mechanismus, nicht ein Geschwindigkeitsvorteil.

## Schnellstart

```bash
# Interaktives Labor (ein p-Bit + 6er-Netz + Live-Kontrolle gegen Theorie)
open p-Bit-Labor.html          # oder einfach im Browser öffnen

# MAX-CUT-Maschine mit Selbsttest
python pbit_maxcut.py

# Faktorisieren (PRNG, Hardware-Rauschen, ohne Rauschen)
python pbit_factor.py --selftest
python pbit_factor.py noise_02.txt --w 3 4 5

# Rauschquelle prüfen (taugt sie als p-Bit-Antrieb?)
python noise_pbit.py noise_02.txt --col 1
```

Abhängigkeiten: Python 3 mit `numpy`. `pbit_factor.py` nutzt bei vorhandenem `gcc` automatisch den schnelleren C-Kern (`pbit_core.c`), sonst einen numpy-Fallback mit identischen Ergebnissen.

## Ergebnisse (Kurzfassung)

Details und Kennzahlen in [`SR_Ergebnisse_Detektion_und_pBit_Rechnen.md`](./SR_Ergebnisse_Detektion_und_pBit_Rechnen.md):

- **Stochastische Resonanz:** SNR-Spitze ca. 30 dB bei optimalem Rauschpegel; Hardware- und PRNG-Rauschen verhalten sich bei diesem Aufbau statistisch nicht unterscheidbar.
- **Faktorisieren mit p-Bits:** Erfolgsrate 99 % bei 3-Bit-Faktoren, fällt auf 0–2 % bei 6 Bit. Kein Geschwindigkeitsvorteil gegenüber Probedivision — gezeigt wird der Mechanismus (Rauschen als Rechenressource), nicht ein praktischer Vorteil.
