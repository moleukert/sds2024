# Lösung aus Moodle übung 4
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def teilaufgabe_a(documents_train, documents_test):
    """
    Implementieren Sie ein Bag of Word Modell zur Umwandlung der 'Lyrics' Spalten im Trainings- und Testdatensatz.
    Nutzen Sie die CountVectorizer Klasse aus der sklearn Bibliothek.

    Der Rückgabewert ist ein 3-tupel bestehend aus:
      vectorizer: Das Bag-of-words Modell (CountVectorizer Instanz aus sklearn)
       X_train:  ein numpy array (integer) in der Struktur ( n x m ). Hier steht n für die Anzahl der Elemente im Trainingsdatensatz
         und m für die Anzahl der abgeleiteten Features. Der Wert 1 (0) bedeutet, dass ein Wort im Dokument (nicht) vorkommt.
       X_test: ein numpy array (integer)  in der Struktur ( n x m ). Hier steht n für die Anzahl der Elemente im Testdatensatz
         und m für die Anzahl der abgeleiteten Features. Der Wert 1 (0) bedeutet, dass ein Wort im Dokument (nicht) vorkommt.

    """

    vectorizer = CountVectorizer(binary=True)
    X_train = vectorizer.fit_transform(documents_train).toarray()
    X_test = vectorizer.transform(documents_test).toarray()

    return vectorizer, X_train, X_test


def teilaufgabe_b(X, y):
    """
    Nutzen Sie den Trainingsdatensatz, um einen Naive Bayes Classifier zu trainieren.
    Rückgabe: ein 2-tuple aus

    priors: ein numpy array mit den a-priori Wahrscheinlichkeiten jeder Klasse
    conds: ein numpy array mit den Wahrscheinlichkeiten jedes Worts in jeder Klasse
           numpy array shape: ( Klassen x Worte )
    """

    m, n = X.shape  # Anzahl von samples (m) und features (n)
    classes = np.unique(y)  # Mögliche Werte im Klassifikationsziel
    n_classes = len(classes)  # Anzahl möglicher Werte im Klassifikationsziel

    # Definition und Berechnung der Wahrscheinlichkeiten im Naive Bayes Classifier
    priors = np.zeros(n_classes)
    conds = np.zeros((n_classes, n))

    # Für jede möglkiche Klasse des Klassifikationstziels (hier 1/0 bzw. True/False)
    for idx, c in enumerate(classes):
        # Selektiere nur die Dokumente der jeweiligen Klasse
        X_c = X[y == c]
        count_docs_in_class = X_c.shape[0]

        # Berechne die Frequenz der Klasse
        priors[idx] = count_docs_in_class / m

        # Berechne die bedingte Wahrscheinlichkeit der möglichen Wörter in der aktuellen Klasse mit Laplace smoothing (+1)
        conds[idx, :] = (X_c.sum(axis=0) + 1) / (count_docs_in_class + m)

    return priors, conds


def teilaufgabe_c(X, classes, priors, conds):
    """
    Nutzen Sie den zuvor trainierte Naive Bayes Klassifikator, um Vorhersagen für einen Datensatz zu treffen.

    Der Rückgabewert ist ein 2-tupel bestehend aus:
       prediction: Ein numpy array mit der binären Klassifikation. (enthält ein boolean je Zeile in X).
       prediction_log_probs: Ein 2D numpy array mit den berechneten Klassenzugehörigkeiten in natürlicher logarithmischer Skala.
                                shape: (Zeilen im Datensatz x mögliche Klassen)
    """

    m = X.shape[0]  # Anzahl der Beispiele (Zeilen in X)
    prediction_log_probs = np.zeros((m, len(classes)))  # Initialisierung der Matrix für Wahrscheinlichkeiten

    for idx, c in enumerate(classes):  # Für jede Klasse im Datensatz
        prior = np.log(priors[idx])  # Logarithmus der a-priori Wahrscheinlichkeiten

        # Berechnet die Logarithmen der bedingten Wahrscheinlichkeiten
        log_prob_feature_1 = np.log(conds[idx, :])

        # Summiere die Produkte aus Feature-Matrix (0/1 je Wort) mit zugehöriger Log-Wahrscheinlichkeit
        term_feature_1 = X @ log_prob_feature_1

        # Summe von Log-Wahrscheinlichkeiten
        prediction_log_probs[:, idx] = prior + term_feature_1

    # Gibt die Klasse mit der höchsten Klassenzugehörigkeit sowie die Klassenzugehörigkeit zurück
    prediction = classes[np.argmax(prediction_log_probs, axis=1)]
    return prediction, prediction_log_probs


if __name__ == "__main__":
    # Laden des Datensatzes
    df_train = pd.read_csv("song_lyrics/train.csv")
    df_test = pd.read_csv("song_lyrics/test.csv")

    # Erstellen einer neuen Spalte mit einem binären Klassifikationslabel
    df_train["Label"] = (df_train["Genre"] == "Metal").astype(int)
    df_test["Label"] = (df_test["Genre"] == "Metal").astype(int)

    # Definition der Klassifikationsziels
    y_test = df_test["Label"].values
    y_train = df_train["Label"].values
    classes = np.unique(y_train)

    # Erstellen eines Bag of Word Modells und Transformation von Training und Testdatensatz
    vectorizer, X_train, X_test = teilaufgabe_a(
        df_train["Lyrics"].values, df_test["Lyrics"].values
    )
    # Trainieren eines Naive Bayes Klassifikators
    priors, conds = teilaufgabe_b(X_train, y_train)

    # Klassifikation eines Datensatzes mit Hilfe des trainierten Modells
    y_pred_test, _ = teilaufgabe_c(X_test, classes, priors, conds)  # Trainingsdatensatz
    y_pred_train, _ = teilaufgabe_c(X_train, classes, priors, conds)  # Testdatensatz

    # Evaluation mittels Metriken
    print(
        "A-priori Wahrscheinlichkeit je Klasse: ", priors
    )  # zu erwarten: ~ [0.6833 0.3166]
    train_accuracy = np.mean(y_pred_train == y_train)
    print(f"Train Accuracy: {train_accuracy:.2f}")  # zu erwarten: >= 0.83

    test_accuracy = np.mean(y_pred_test == y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")  # zu erwarten: >= 0.81
    # Weitere manuelle Evaluation des Klassifikators mit Texten aus Nutzereingaben
    while True:
        user_input = input("Enter some text (or press Enter to exit): ")
        if user_input == "":
            break
        else:
            X_user = vectorizer.transform(pd.Series([user_input])).toarray()
            y_user, y_log_probs_user = teilaufgabe_c(X_user, classes, priors, conds)

            # Umrechnung von logarithmischen Klassenzugehörigkeiten in Wahrscheinlichkeiten
            max_log_probs = np.max(y_log_probs_user, axis=1, keepdims=True)
            # Vermeiden von overflow, relative Größenordnung behalten
            probs = np.exp(y_log_probs_user - max_log_probs)
            # Normalisierung zu Wahrscheinlichkeiten
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            probability = np.max(probs)

            is_metal = bool(y_user[0])
            str_klassenzugehoerigkeit = f"Logar. Klassenzugehörigkeiten = {y_log_probs_user[0][0]:.5} {y_log_probs_user[0][1]:.5}"

            print(
                f"{'' if is_metal else 'Kein '}Metal (Konfidenz: {probability:.0%}, {str_klassenzugehoerigkeit})"
            )
