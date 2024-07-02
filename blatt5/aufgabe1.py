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
    """


def teilaufgabe_b(samples_a, samples_b):
    """
    Führen Sie einen T-Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie die Differenz der Mittelwerte, den p-value und das Testergebnis (boolean) zurück.
    """

    # Implementieren Sie hier Ihre Lösung.
    mean_diff = None
    p_value = None
    decision = None

    """
    Interpretation: Formulieren Sie hier ihre Antwort. 
    
    """

    return mean_diff, p_value, decision


def teilaufgabe_c(samples_a, samples_b):
    """
    Führen Sie den Mann-Whitney U Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie den p-value und das Testergebnis (boolean) zurück.
    """

    # Implementieren Sie hier Ihre Lösung.
    p_value = None
    decision = None

    """
    Argumente: Formulieren Sie hier ihre Antwort. 
    """
    return p_value, decision


if __name__ == "__main__":
    samples_a = np.array([0.24, 0.22, 0.20, 0.25], dtype=np.float64)
    samples_b = np.array([0.2, 0.19, 0.22, 0.18], dtype=np.float64)

    print("Teilaufgabe b)")

    mean_diff, p_value, decision = teilaufgabe_b(samples_a, samples_b)
    print(f"{mean_diff=}")  # ~ 0.03
    print(f"{p_value=}")  # ~ 0.038
    print(f"{decision=}")  # ~ True

    print()
    print("Teilaufgabe c)")
    p_value, decision = teilaufgabe_c(samples_a, samples_b)
    print(f"{p_value=}")  # ~ 0.054
    print(f"{decision=}")  # ~ False
