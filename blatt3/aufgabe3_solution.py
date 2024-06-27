# Lösung aus Moodle übung 5
from os.path import realpath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def teilaufgabe_a():
    """
    Rückgabewerte:
    sides: numpy array (integer, 12 Elemente), die Werte der Würfelseiten
    side_probabilities: numpy array (float, 12 Elemente), die Wahrscheinlichkeit jeder Würfelseite
    expected_value: float, der Erwartungswert der Zufallsvariable "Würfelergebnis"
    cdf: Kumulative Wahrscheinlichkeitsverteilungsfunktion
    """
    sides = np.array([1, 1, 1, 1, 2, 2, 2, 4, 4, 8, 16, 32])
    side_probabilities = np.ones_like(sides) / len(sides)
    expected_value = np.sum(sides * side_probabilities)

    def cdf(X: float):
        mask = sides <= X
        return np.sum(side_probabilities[mask])

    return sides, side_probabilities, expected_value, cdf


def teilaufgabe_b():
    """
    Rückgabewerte:
    fig: die matplotlib figure
    sample_mean: float, der gesuchte mean der Würfelergebnisse im Datensatz
    """
    spieler_name = 1
    tisch_name = "B"
    change_time = pd.Timestamp("2024-03-27 21:00:00")

    df = pd.read_csv("casino.csv", index_col=False)
    df["zeit"] =pd.to_datetime(df["zeit"])

    df_b = df[(
            (df.spieler == spieler_name) & (df.tisch == tisch_name) & (df.zeit >= change_time)
    )]

    ergebnis_counts = df_b.ergebnis.value_counts().sort_index()
    unique_sides = ergebnis_counts.index

    y1 = ergebnis_counts.values / np.sum(ergebnis_counts.values)

    # Konvertierung zu string für kategorische Skala
    X = unique_sides.astype(str)

    fig, ax = plt.subplots()
    ax.bar(X, y1, width=0.2, label="Spieler")
    ax.set_xlabel(f"Würfelergebnis von Spieler {spieler_name}, Tisch {tisch_name}")
    ax.set_ylabel("Relative Häufigkeit")

    sample_mean = df_b.ergebnis.mean()

    """
    Der Sample Mean ist bei Spieler 1 an Tisch B ab 21 Uhr mit einem Wert von über 10 im Vergleich zum Erwartungswert
    von circa 6.167 unerwartet hoch, was auf einen manipulierten Würfel vermuten lässt.
    Bei einem fairen Würfel ist die kumulierte Wahrscheinlichkeit für Ergebnisse <= 2 circa 0.583, in diesem Fall liegt
    er lediglich bei circa 0.369, das heißt die niedrigen Zahlen sind deutlich weniger Häufig aufgetreten.
    """

    return fig, sample_mean


def teilaufgabe_c(expected_value_fair, spieler_name=1, tisch_name="B"):
    """
    Rückgabewert:
    fig: die matplotlib figure
    """
    df = pd.read_csv("casino.csv", index_col=False)
    df["zeit"] = pd.to_datetime(df["zeit"])
    change_time = pd.Timestamp("2024-03-27 21:00:00")

    df_b = df[(df.spieler == spieler_name) & (df.tisch == tisch_name)]

    sample_means = df_b["ergebnis"].expanding().mean()

    fig, ax = plt.subplots()
    ax.plot(
        df_b.zeit,
        sample_means,
        label=f"Spieler {spieler_name}, Tisch {tisch_name}",
    )

    ax.axvline(
        change_time,
        ymin=0,
        ymax=1,
        label="Zeitpunkt",
        color="grey",
        linestyle=":",
    )

    ax.axhline(
        expected_value_fair,
        xmin=0,
        xmax=1,
        label="Fairer Würfel",
        color="grey",
        linestyle="--",
    )

    ax.set_ylabel("Mean des Würfelergebnisses")
    ax.set_xlabel("Uhrzeit")
    ax.legend()

    """
    Wenn wir die Visualisierungen der Ergebnisse von Spieler 1 und 2 vergleichen, fällt direkt eine starke Tendenz, ab 
    dem Zeitpunkt auf, ab dem bei Spieler 1 der Austausch vermutet wird. Sein Sample Mean steigt kontinuierlich, sprich
    er würfelt tendenziell immer höhere Zahlen. Wohingegen der Samples Mean von Spieler 2 nah am Erwartungswert bleibt.
    Das Gesetz der großen Zahlen besagt, dass die beobachtete Häufigkeit, mit der ein Zufallsereignis eintritt, sich
    dem rechnerischen Erwartungswert weiter annähert, so häufiger das Experiment durchgeführt wird. Die Ergebnisse von
    Spieler 2 treffen auf diese Beschreibung zu, bei Spieler 1 ist dies nicht zu beobachten, was wiederum den Einsatz
    eines manipulierten Würfels naheliegt.
    """

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
