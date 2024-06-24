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
    fig, ax = plt.subplots()
    expected_mean = sum(range(1, 7)) / 6 * 2  # 3.5 für einen Würfel -> 7.0 für zwei
    experiment = pd.DataFrame(index=range(1, 10001), columns=['Ergebnis', 'Sample Mean'])  # Dataframe für Würfeln
    n, m = 200, 10000
    # Zufälliges Würfeln mit zwei fairen, sechsseitigen Würfeln

    for i in range(1, m+1):
        experiment['Ergebnis'][i] = np.sum(np.random.randint(1, 7, (n, 2)))
        experiment['Sample Mean'][i] = experiment['Ergebnis'][i] / n

    ax.hist(experiment['Sample Mean'], edgecolor='k')
    ax.set_xticks(range(2, 13, 1))
    ax.axvline(
        expected_mean,
        ymin=0,
        ymax=1,
        label="Expected Mean der beiden Würfel",
        color="grey",
        linestyle=":",
    )
    ax.set_title('Häufigkeitsverteilung des Sample Means für alle 10.000 Durchläufe')
    ax.set_ylabel("Absolute Häufigkeit des Sample Means")
    ax.set_xlabel("Sample Mean der Würfelsumme aus zwei fairen, sechsseitigen Würfeln")
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(linestyle='dashed')
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

    m = 10000  # Wiederholungen
    n = 50  # mit je zufälligen Samples

    hyperSimulation = pd.DataFrame(index=range(1, m+1), columns=['Summen', 'Mittelwerte'])
    poissonSimulation = pd.DataFrame(index=range(1, m+1), columns=['Summen', 'Mittelwerte'])

    # Simulation der Verteilungen durch 10.000 Wiederholungen
    for idx in range(1, m+1):
        # 1) Hypergeometrische Verteilung h(x|49; 6; 6)
        hyperSamples = np.random.hypergeometric(6, 49 - 6, 6, n)
        hyperSum = np.sum(hyperSamples)
        hyperMean = np.mean(hyperSamples)
        hyperSimulation.loc[idx] = [hyperSum, hyperMean]

        # 2) Poisson Verteilung p(x) lambda = 3,17
        poissonSamples = np.random.poisson(3.17, n)
        poissonSum = np.sum(poissonSamples)
        poissonMean = np.mean(poissonSamples)
        poissonSimulation.loc[idx] = [poissonSum, poissonMean]

    # Tuple für Daten, Titel, xLabel, yLabel und Farbe
    config = (
        (hyperSimulation['Summen'], 'Summen der hypergeometrischen Verteilung', 'Summe einer Wiederholung',
         'Absolute Häufigkeit', 'cornflowerblue'),
        (hyperSimulation['Mittelwerte'], 'Mittelwerte der hypergeometrischen Verteilung',
         'Mittelwert einer Wiederholung', 'Absolute Häufigkeit', 'lightcoral'),
        (poissonSimulation['Summen'], 'Summen der Poisson Verteilung', 'Summe einer Wiederholung',
         'Absolute Häufigkeit', 'limegreen'),
        (poissonSimulation['Mittelwerte'], 'Mittelwerte der Poisson Verteilung', 'Mittelwert einer Wiederholung',
         'Absolute Häufigkeit', 'mediumslateblue')
    )

    # Schleife über die Konfiguration mit den Daten, um jeweils ein Histogramm zu erzeugen
    for idx, (data, title, xlabel, ylabel, color) in enumerate(config, start=2):
        fig = plt.figure(idx)
        plt.hist(data, bins=40, color=color, edgecolor='k', zorder=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(linestyle='dashed', zorder=0)
        figures.append(fig)
    '''
    Interpretation: Alle beobachteten Verteilungen mit stimmen mit der Erwartung nach dem zentralen Grenzsatz überein.
    Die Summen und Mittelwerte der hypergeometrischen Verteilung zeigen fast eine perfekte Normalverteilung, da
    beide Verteilungen um den Mittelwert die höchsten Häufigkeiten aufweisen und diese Häufigkeiten in beide Richtungen
    symmetrisch abnehmen.
    Bei den Summen und Mittelwerten der Poisson Verteilung verhält es sich ähnlich, allerdings gibt es hier bei manchen
    Werten ein paar Ausreißer d.h. diese Werte sind mit höherer Häufigkeit zu beobachten, als es unter einer 
    Normalverteilung angenommen wird.
    Trotz der verschiedenen ursprünglichen Verteilungen konvergieren beide Summen und Mittelwerte bei der großen Anzahl
    von Stichproben zur Form einer Normalverteilung und zeigen das erwartete Verhalten gemäß zentralem Grenzwertsatz.
    '''
    return figures


def teilaufgabe_c():
    """
    Rückgabewerte:
    figures: Eine Liste aller matplotlib Figures
    """
    figures = []

    mValues = [10, 100, 1000, 10000]
    n = 6
    targetAttributes = ['HP', 'Attack', 'Speed']
    pokemon = pd.read_csv('pokemon.csv')
    colors = ['limegreen', 'lightcoral', 'cornflowerblue']

    for idx, attribute in enumerate(targetAttributes):  # Für jedes Attribut
        fig, axes = plt.subplots(len(mValues), 1, sharex='all')
        fig.suptitle(attribute)
        for jdx, mValue in enumerate(mValues): # Für jeden M Wert
            sampleMeans = [np.mean(pokemon[attribute].sample(n)) for _ in range(mValue)]  # von m-Mal sampeln 0 bis m-1

            zSampleMeans = (sampleMeans - np.mean(sampleMeans)) / np.std(sampleMeans)  # Z-Transformation durchführen

            axes[jdx].hist( # Histogramm mit den relativen Häufigenkeiten hinzufügen
                zSampleMeans,
                density=True,
                bins=40, color=colors[idx], edgecolor='k', label=f'{mValue = }', zorder=2)
            x = np.linspace(-3, 3, 1000)
            y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
            axes[jdx].plot(x, y, linestyle='dashed', color='orange')  # Standardnormalverteilung hinzufügen
            axes[jdx].grid(linestyle='dashed', zorder=0)
            axes[jdx].legend()

        plt.xlabel('z-transformierter Sample Mean')
        plt.ylabel('relative Häufigkeit')
        plt.tight_layout()

        figures.append(fig)
    '''
    Interpretation: Bei m=10 zeigt die Verteilung eine grobe Abweichung von der Normalverteilung, dies ist zu erwarten,
    da der Stichprobenumfang sehr gering ist.
    Mit m=100 bessert sich die Verteilung, wie erwartet. Da der Stichprobenumfang wächst nähert sich die Verteilung, 
    einer Normalverteilung, allerdings gibt nichtsdestotrotz noch ein paar unerwartete Ausreißer.
    Bei m=1000 ist der Stichprobenumfang so groß, dass die Verteilung einer Normalverteilung sehr nahe kommt. Es gibt
    zwar noch hier und dort ein paar dezente Abweichungen, dennoch kommt der zentrale Grenzwertsatz hier sichtlich zum 
    tragen. Bei einem Stichprobenumfang von m=10000 sieht man eine nahezu perfekte Normalverteilung durch die Verteilung,
    dieses letzte Diagramm der Verteilung bestätigt den zentralen Grenzwertsatz. Die Verteilung der transformierten
    Mittelwerte aus der Stichprobe näheren sich einer Normalverteilung äußerst gut an.
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
