import numpy as np
from scipy.stats import t, mannwhitneyu


# -------------------------------
# Abgabegruppe: Gruppe 10
# Personen: Alisha Vaders, Moritz Leukert, Yann-Cédric Gagern
# HU-Accountname: vadersal, leukertm, gagernya
# -------------------------------


def teilaufgabe_a():
    return """
    Formulieren Sie hier beide Testhypothesen.
    Nullhypothese H0: muNEU <= muALT
    Die durchschnittliche Reaktionszeit der neuen App Version A hat sich im Vergleich zu Version B nicht signifikant
    verschlechtert.
    Alternativhypothese HA: muNEU > muALT
    Die durchschnittliche Reaktionszeit der neuen App Version A hat sich im Vergleich zu Version B signifikant 
    verschlechtert.
    """


def teilaufgabe_b(samples_a, samples_b):
    """
    Führen Sie einen T-Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie die Differenz der Mittelwerte, den p-value und das Testergebnis (boolean) zurück.
    """

    n, m = len(samples_a), len(samples_b)  # Anzahl der Samples
    mean_diff = np.mean(samples_a) - np.mean(samples_b)  # Differenz der Mittelwerte
    # Varianz der Zweistichproben mit Bessel-Korrektur
    s_squared = ((n - 1) * np.var(samples_a, ddof=1) + (m - 1) * np.var(samples_b, ddof=1)) / (n + m - 2)
    t_value = mean_diff / np.sqrt(s_squared * (1/n + 1/m))  # t-Wert
    p_value = t.cdf(-t_value, (n + m - 2))  # p-Wert bei einseitigem Test
    decision = p_value < 0.05  # p-Wert < Signifikanzniveau → H0 kann zugunsten HA verworfen werden
    """
    Interpretation: Formulieren Sie hier ihre Antwort. 
    Die Wahrscheinlichkeit, dieses oder eines extremeren Ergebnisses liegt bei circa 3,8%. Damit liegt sie unter dem 
    Signifikanzniveau Alpha von 5%, die Wahrscheinlichkeit, dass zufällig dieses oder ein extremeres Ergebnis auftritt 
    und so kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden.
    Es liegt signifikante Evidenz vor, dass sich die durchschnittliche Reaktionszeit der neuen App Version A 
    im Vergleich zur alten Version B verschlechtert hat.
    """

    return mean_diff, p_value, decision


def teilaufgabe_c(samples_a, samples_b):
    """
    Führen Sie den Mann-Whitney U Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie den p-value und das Testergebnis (boolean) zurück.
    """

    p_value = mannwhitneyu(samples_a, samples_b, alternative='greater').pvalue  # p-Wert bei einseitigem Test
    decision = p_value < 0.05  # p-Wert < Signifikanzniveau → H0 kann zugunsten HA verworfen werden

    """
    Argumente: Formulieren Sie hier ihre Antwort.
    Die Nullhyptothese kann nicht zugunsten der Alternativhypothese verworfen werden. Da der p-Wert >= Signifikanzniveau
    mit 0,054 > 0,05 ist. Es liegt keine signifikante Evidenz vor, dass die durchschnittliche Reaktionszeit der neuen
    App Version A sich im Vergleich zur alten Version B verschlechtert hat.
    Der T-Test geht von einer Normalverteilung der Reaktionszeit aus, so wie wir es ebenfalls in der Aufgabe tun.
    Der Mann-Whitney-U-Test geht dagegen nicht von einer Normalverteilung aus, welche hier nicht vorliegt.
    Aufgrund der Verteilung ist in diesem speziellen Fall entsprechend dem T-Test mehr Glauben zu schenken.
    """
    return p_value, decision


if __name__ == "__main__":
    samples_a = np.array([0.24, 0.22, 0.20, 0.25], dtype=np.float64)
    samples_b = np.array([0.2, 0.19, 0.22, 0.18], dtype=np.float64)

    print("Teilaufgabe b)")

    mean_diff, p_value, decision = teilaufgabe_b(samples_a, samples_b)
    print(f"{mean_diff=:.2f}")  # ~ 0.03
    print(f"{p_value=:.3f}")  # ~ 0.038
    print(f"{decision=}")  # ~ True

    print()
    print("Teilaufgabe c)")
    p_value, decision = teilaufgabe_c(samples_a, samples_b)
    print(f"{p_value=:.3f}")  # ~ 0.054
    print(f"{decision=}")  # ~ False
