# -------------------------------
# Abgabegruppe: Gruppe 10
# Personen: Alisha Vaders, Moritz Leukert, Yann-Cédric Gagern
# HU-Accountname: vadersal, leukertm, gagernya
# -------------------------------
from os.path import realpath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates

def teilaufgabe_a():
    """
    Rückgabewerte:
    sides: numpy array (integer, 12 Elemente), die Werte der Würfelseiten
    side_probabilities: numpy array (float, 12 Elemente), die Wahrscheinlichkeit jeder Würfelseite
    expected_value: float, der Erwartungswert der Zufallsvariable "Würfelergebnis"
    cdf: Kumulative Wahrscheinlichkeitsverteilungsfunktion
    """

    sides = np.array([1, 1, 1, 1, 2, 2, 2, 4, 4, 8, 16, 32], dtype=int)
    side_probabilities = np.full(shape=len(sides), fill_value=1/len(sides), dtype=float)
    expected_value = np.sum(sides * side_probabilities)

    def cdf(X: float):
        cumu_prob = 0
        for side, prob in zip(sides, side_probabilities):
            if side <= X:
                cumu_prob += prob

        return cumu_prob

    return sides, side_probabilities, expected_value, cdf


def teilaufgabe_b():
    """
    Rückgabewerte:
    fig: die matplotlib figure
    sample_mean: float, der gesuchte mean der Würfelergebnisse im Datensatz
    """
    fig, ax = plt.subplots()

    casino = pd.read_csv('casino.csv')
    casino['zeit'] = pd.to_datetime(casino['zeit'])
    casino = casino[(casino['zeit'].dt.hour >= 21) &
                    (casino['tisch'] == 'B') &
                    (casino['spieler'] == 1)]

    sample_mean = np.mean(casino['ergebnis'])

    count_results = casino['ergebnis'].value_counts()
    rel_proba = count_results / casino['ergebnis'].count()
    rel_proba.sort_index(inplace=True)

    rel_proba.plot(kind='bar', xlabel='Ergebnis', ylabel='Relative Häufigkeiten', edgecolor='k', zorder=2) # theoretisch nicht alle möglichen Ergebnisse dabei
    plt.title('Relative Häufigkeiten der Würfelergebnisse für Spieler 1 an Tisch B nach 21:00 Uhr', fontsize=10)
    plt.xticks(rotation=0)
    plt.grid(zorder=0)

    return fig, sample_mean


def teilaufgabe_c(expected_value_fair, spieler_name=1, tisch_name="B"):
    """
    Rückgabewert:
    fig: die matplotlib figure
    """

    fig, ax = plt.subplots()

    casino = pd.read_csv('casino.csv')
    casino['zeit'] = pd.to_datetime(casino['zeit'])
    casino = casino[(casino['tisch'] == tisch_name) &
                    (casino['spieler'] == spieler_name)]
    casino.sort_values(by=['zeit'], inplace=True)
    casino['sample_mean'] = casino['ergebnis'].expanding().mean()

    plt.plot(casino['zeit'], casino['sample_mean'], label=f'Sample Mean der Würfelergebnisse des Spielers {spieler_name}')
    plt.axhline(y=expected_value_fair, linestyle='dashed', color='grey', label='Erwartungswert')
    plt.axvline(pd.to_datetime('2024-03-27 21:00:00'), linestyle='dotted', color='grey', label='Zeitpunkt des möglichen Würfeltauschs - 21 Uhr')

    plt.xlabel('Zeitpunkt')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %Hh'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.ylabel('Sample Mean')
    plt.title(f'Zeitlicher Ablauf der Würfelergebnisse von Spieler {spieler_name} an Tisch {tisch_name}', fontsize=10)
    plt.legend(loc='best')
    plt.grid(zorder=0)

    return fig


if __name__ == "__main__":
    figures = []

    sides, side_probabilities, expected_mean, cdf = teilaufgabe_a()

    # Test der Ergebnisse aus Teilaufgabe (a)
    print("Teilaufgabe (a) :")
    assert len(sides) == 12, "'sides' muss 12 Elemente haben."
    assert len(side_probabilities) == 12, "'side_probabilities' muss 12 Elemente haben."
    assert np.isclose(
        np.sum(side_probabilities), 1.0
    ), "Die Summe aller Wahrscheinlichkeiten muss 1 sein."
    print(f"{expected_mean=}")  # ~ 6.167
    print(f"{cdf(0)=}")  # ~ 0.0
    print(f"{cdf(1)=}")  # ~ 0.333
    print(f"{cdf(2)=}")  # ~ 0.583
    print(f"{cdf(42)=}")  # ~ 1.0

    # Visualisierung der kumulativen Wahrscheinlichkeitsfunktion
    fig, ax = plt.subplots()
    X = np.arange(0, 33, 0.01)
    ax.plot(X, [cdf(x) for x in X])
    ax.set_xlabel("Mögliches Würfelergebnis")
    ax.set_ylabel("Kumulative Wahrscheinlichkeit")
    ax.set_xlim(0, 33)
    ax.grid()

    figures.append(fig)

    fig, sample_mean = teilaufgabe_b()

    figures.append(fig)

    print()
    print("Teilaufgabe (b) :")
    print(f"{sample_mean=}")  # ~ 10.754

    fig = teilaufgabe_c(expected_mean)
    figures.append(fig)

    fig = teilaufgabe_c(expected_mean, spieler_name=2, tisch_name="B")
    figures[-1].axes[0].sharey(fig.axes[0])
    figures.append(fig)

    # Save the figures to a multi-page PDF
    pdf_path = "aufgabe3_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print()
    print(f"Figures saved to {realpath(pdf_path)}")
