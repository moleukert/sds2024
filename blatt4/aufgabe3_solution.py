# Lösung aus Moodle übung 6
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
    sides = np.array([1, 2, 3, 4, 5, 6])
    expected_mean = np.mean(np.add.outer(sides, sides))

    first = np.random.choice(sides, size=(10000, 200))
    second = np.random.choice(sides, size=(10000, 200))
    samples = first + second

    sample_means = np.mean(samples, axis=1)

    fig, ax = plt.subplots()
    ax.hist(sample_means, bins=50, edgecolor='k')
    ax.axvline(
        expected_mean,
        ymin=0,
        ymax=1,
        label='Expected Mean der beiden Würfel',
        color='grey',
        linestyle='--'
    )
    ax.set_title('Häufigkeitsverteilung des Sample Means für alle 10.000 Durchläufe')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('abs. Häufigkeit')
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(linestyle='dashed')
    '''
    Interpretation: Die beobachteten Sample Means scheinen um den Erwartungswert von 7 normalverteilt zu sein. 
    Dies entspricht der Erwartung, dass nach dem zentralen Grenzwertsatz der Sample Mean bei hoher Stichprobengröße 
    annähernd normalverteilt ist.
    '''
    return fig, expected_mean


def teilaufgabe_b():
    """
    Rückgabewerte:
    figures: Eine Liste aller matplotlib Figures
    """
    figures = []
    n = 50  # Größe einer Stichprobe
    m = 10000  # Anzahl von Stichproben

    # Funktionen zur Generierung von Stichproben
    stichprobe_lotto = np.random.hypergeometric(6, 49 - 6, 6, size=(m, n))
    stichprob_bundesliga = np.random.poisson(3.17, size=(m, n))

    for stichprobe, label_dataset, distribution in (
        (stichprobe_lotto, 'Anzahl Richtiger in 6 aus 49', 'Hypergeometrischen Verteilung'),
        (stichprob_bundesliga, 'Tore in Bundesligaspiel', 'Poisson Verteilung')
    ):
        for fn_statistic, label_statistic, color in (
            (np.mean, 'Mittelwert', 'limegreen'),
            (np.sum, 'Summe', 'dodgerblue')
        ):
            fig, ax = plt.subplots()
            sample_stats = fn_statistic(stichprobe, axis=1)
            ax.hist(sample_stats, bins=200, color=color)
            ax.set_title(distribution)
            ax.set_xlabel(f'{m = }, {label_statistic} von {label_dataset}')
            ax.set_ylabel('abs. Haufigkeit')
            ax.set_axisbelow(True)
            ax.grid(linestyle='dashed')

            figures.append(fig)
    '''
    Interpretation: Die beobachteten Summen und Mittelwerte scheinen normalverteilt zu sein, obwohl die ursprünglichen 
    Verteilungen Hypergeometrisch bzw. Poisson Verteilungen waren. Dies entspricht der Erwartung, dass nach dem 
    zentralen Grenzwertsatz Mittelwert und Summe bei ausreichend hoher Stichprobengröße annähernd normalverteilt sind. 
    '''
    return figures


def teilaufgabe_c():
    """
    Rückgabewerte:
    figures: Eine Liste aller matplotlib Figures
    """
    figures = []
    pokemon = pd.read_csv('pokemon.csv')
    attributes = ['HP', 'Attack', 'Speed']
    m_values = [10, 100, 1000, 10000]
    n = 6
    colors = ['limegreen', 'lightcoral', 'cornflowerblue']

    for attribute, color in zip(attributes, colors):
        fig, axes = plt.subplots(len(m_values), 1, figsize=(6, 7), sharex='all')
        fig.suptitle(f'Histogramme für das Attribut {attribute}')
        for m, ax in zip(m_values, axes):
            stichproben = np.random.choice(pokemon[attribute], size=(m, n))
            sample_statistic = np.mean(stichproben, axis=1)

            # Z transformation
            sample_statistic -= np.mean(sample_statistic)
            std = np.std(sample_statistic)
            sample_statistic /= std if std != 0 else 1

            ax.hist(sample_statistic, bins=200, density=True, color=color, label=f'{m = }')
            x = np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 0.01)
            ax.plot(
                x,
                (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2),
                label='Standard Normalverteilung', linestyle='dashed', color='orange'
            )
            # ax.legend()
            ax.set_xlabel(f'{m = }, z-transformierter Sample Mean von {attribute}')
            ax.set_ylabel('rel. Haufigkeit')

        figures.append(fig)
    '''
    Interpretation: Die beobachteten Mittelwerte der Stichproben scheinen annährend normalverteilt zu sein. Dies ist bei 
    zu geringer Stichprobenanzahl noch nicht erkennbar, jedoch bei ausreichend großer Anzahl. Dies entspricht der 
    Erwartung, dass nach dem zentralen Grenzwertsatz der Stichproben-Mittelwert bei hoher Stichprobengröße annähernd 
    normalverteilt ist.
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
