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


def teilaufgabe_a():
    """
    Rückgabewerte:
    fig: die matplotlib figure
    expected_mean: float
    """
    # TODO Implementieren Sie hier Ihre Lösung
    fig, ax = plt.subplots()
    expected_mean = sum(range(1,7)) / 6 * 2
    experiment = pd.DataFrame(index=range(1,10001), columns=['Ergebnis', 'Sample Mean'])

    experiment['Ergebnis'] = np.random.randint(1,7,len(experiment)) + np.random.randint(1,7,len(experiment))
    sample_means = experiment['Ergebnis'].expanding().mean()

    ax.hist(
        sample_means,
        bins=50,
        alpha=0.80,
        edgecolor='k',
        label=f""
    )

    ax.axvline(
        expected_mean,
        ymin=0,
        ymax=1,
        label="Expected Mean der beiden Würfel",
        color="grey",
        linestyle=":",
    )
    ax.set_title('Häufigkeitsverteilung des Sample Means für alle 10.000 Durchläufe', fontsize=10)
    ax.set_ylabel("Absolute Häufigkeit des Sample Means")
    ax.set_xlabel("Sample Mean der Würfelsumme aus zwei fairen, sechsseitigen Würfeln")
    ax.legend()
    '''
    Interpretation: Das Histogramm zeigt, dass sich die wirkliche Verteilung um den erwarteten Mittelwert verteilt.
    Laut des zentralen Grenzwertsatzes sollte sich der Mittelwert unabhängiger und identisch verteilter Zufallsvariablen
    bei einer beliebigen Verteilung mit zunehmendem Stichprobenumfang der Normalverteilung annähren.
    Die hier beobachtete Verteilung bestätigt diese Aussage und somit den zentralen Grenzwertsatz, da sie einer 
    Normalverteilung, welche um den erwarteten Mittelwert verteilt ist, nahe kommt. 
    '''
    return fig, expected_mean


def teilaufgabe_b():
    """
    Rückgabewerte:
    figures: Eine Liste aller matplotlib Figures
    """
    figures = []

    # TODO Implementieren Sie hier Ihre Lösung

    '''
    Interpretation:
    '''
    return figures


def teilaufgabe_c():
    """
    Rückgabewerte:
    figures: Eine Liste aller matplotlib Figures
    """
    figures = []

    # TODO Implementieren Sie hier Ihre Lösung

    '''
    Interpretation:
    '''
    return figures


if __name__ == "__main__":
    figures = []

    fig, expected_mean = teilaufgabe_a()
    figures.append(fig)
    print(f"{expected_mean=}")  # ~7.0

    figures_b = teilaufgabe_b()
    figures.extend(figures_b)

    figures_c = teilaufgabe_c()
    figures.extend(figures_c)

    # Save the figures to a multi-page PDF
    pdf_path = "aufgabe3_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            if fig is None:
                continue

            pdf.savefig(fig)
            plt.close(fig)

    print()
    print(f"Figures saved to {realpath(pdf_path)}")
