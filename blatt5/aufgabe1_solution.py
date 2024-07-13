# Lösung aus Moodle übung 7
import numpy as np
from scipy.stats import t, mannwhitneyu

def teilaufgabe_a():
    return """
    Nullhypothese H0: (besser oder gleich schnell)
    Die Reaktionszeit der neuen App-Version (Gruppe A) ist kleiner oder gleich als die Reaktionszeit der bisherigen 
    App-Version (Gruppe B).
    
    Alternativhypothese HA: (langsamer)
    Die Reaktionszeit der neuen App-Version (Gruppe A) ist größer als die Reaktionszeit der bisherigen 
    App-Version (Gruppe B).
    """


def teilaufgabe_b(samples_a, samples_b):
    """
    Führen Sie einen T-Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie die Differenz der Mittelwerte, den p-value und das Testergebnis (boolean) zurück.
    """

    # Schritt 1: Berechnen der Stichprobenmittelwerte und -varianzen mit Bessel Korrektur.
    mean_A, mean_B = np.mean(samples_a), np.mean(samples_b)
    var_A, var_B = np.var(samples_a, ddof=1), np.var(samples_b, ddof=1)
    mean_diff = np.abs(mean_A - mean_B)

    # Schritt 2: Berechnung des gewichteten Mittels der korrigierten Stichprobenvarianz
    n_A, n_B = len(samples_a), len(samples_b)
    dof = n_A + n_B - 2
    pooled_var = ((n_A - 1) * var_A + (n_B - 1) * var_B) / dof  # -2 = ddof wegen Bessel Korrektur

    # Schritt 3: Berechnung der t-Statistik
    t_stat = (mean_A - mean_B) / np.sqrt(pooled_var * ((1 / n_A) + (1 / n_B)))

    # Schritt 4: Ermitteln des p-Wertes
    p_value = 1 - t.cdf(t_stat, dof)

    # Schritt 5: Entscheidung nach Signifikanzlevel alpha = 0.05
    decision = p_value < 0.05

    """
    Interpretation: 
    Die Nullhypothese kann zugunsten der Alternativhypothese zurückgewiesen werden. Aus dem p-Wert folgt: Die 
    Wahrscheinlichkeit, dass die Reaktionszeit der neuen App-Version (Gruppe A) kleiner oder gleich als die 
    Reaktionszeit der bisherigen App-Version (Gruppe B) ist und die beobachteten Unterschiede zufällige
    Schwankungen sind, liegt bei 3.8%. 
    """

    return mean_diff, p_value, decision


def teilaufgabe_c(samples_a, samples_b):
    """
    Führen Sie den Mann-Whitney U Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie den p-value und das Testergebnis (boolean) zurück.
    """

    _, p_value = mannwhitneyu(samples_a, samples_b, alternative="greater")  # einseitiger Test
    decision = p_value < 0.05  # alpha 5%

    """
    Argumente: 
    Die Nullhypothese kann nicht zurückgewiesen werden. Der T-Test (T) ist ein parametrischer, und der 
    Mann-Whitney-U-Test (U) ein nichtparametrischer Test. Konkret nimmt T an, dass die Stichproben normalverteilt sind.
    Sofern diese Annahme zutrifft, ist T mächtiger und demnach mehr Glauben zu schenken. Wenn die Annahme nicht 
    zutrifft, ist das Ergebnis aus T unzuverlässig und U ist als nicht-parametrischer Test vorzuziehen. 
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
