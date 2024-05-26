# -------------------------------
# Abgabegruppe: Gruppe 10
# Personen: Alisha Vaders, Moritz Leukert, Yann-Cédric Gagern
# HU-Accountname: vadersal, leukertm, gagernya
# -------------------------------
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

    # Implementieren Sie hier Ihre Lösung
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(documents_train)
    X_test = vectorizer.fit_transform(documents_test)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    X_train[X_train>1] = 1
    X_test[X_test>1] = 1
    np.savetxt("xtest.txt",X_test[0],delimiter=",",fmt="%d")
    return vectorizer, X_train, X_test


def teilaufgabe_b(X, y):
    """
    Nutzen Sie den Trainingsdatensatz, um einen Naive Bayes Classifier zu trainieren.
    Rückabe: ein 2-tuple aus

    priors: ein numpy array mit den a-priori Wahrscheinlichkeiten jeder Klasse
    conds: ein numpy array mit den Wahrscheinlichkeiten jedes Worts in jeder Klasse
           numpy array shape: ( Klassen x Worte )
    """

    # Implementieren Sie hier Ihre Lösung
    sum = 0
    for i, x in enumerate(y):
        sum += x

    priors = np.array([1-(sum/len(y)), sum/len(y)])
    # go through each X[i] (this is one song) and 

    x_n, x_m = X.shape
    conds = np.zeros((2,x_m))

    for f in range(2):
        for j in range(x_m):
            sum = 0
            count = 0
            for i in range(x_n):              
                if( y[i] == f):
                    sum+=X[i][j]
                    count+=1
            # laplace glättung
            sum += 1
            count += x_n * 1
            # berechnung einer einzelnen conditional probability
            conds[f][j] = sum/count
 
    return priors, conds


def teilaufgabe_c(X, classes, priors, conds):
    """
    Nutzen Sie den zuvor trainierte Naive Bayes Klassifikator, um Vorhersagen für einen Datensatz zu treffen.

    Der Rückgabewert ist ein 2-tupel bestehend aus:
       prediction: Ein numpy array mit der binären Klassifikation. (enthält ein boolean je Zeile in X).
       prediction_log_probs: Ein 2D numpy array mit den berechneten Klassenzugehörigkeiten in natürlicher logarithmischer Skala.
                                shape: (Zeilen im Datensatz x mögliche Klassen)
    """
    # Implementieren Sie hier Ihre Lösung
    # TODO: check for errors and use classes
    x_n, x_m = X.shape
    prediction = np.zeros(x_n)
    prediction_log_probs = np.zeros(x_n)
    for i in range(x_n):
        # set a priori (no, yes)
        prob_n = priors[0]
        prob_y = priors[1]
        # multiply with conditional
        for j in range(x_m):
            # check if we have the feature or not
            t = abs(X[i][j] -1)
            
            # conditional for each class
            prob_n *= abs(t - conds[0][j])
            prob_y *= abs(t - conds[1][j])
        # catch 0
        if(prob_n == 0):
            prob_n = 6e-323
        if(prob_y == 0):
            prob_y = 6e-323
        
        # compare probs
        prediction[i] = (prob_y > prob_n)
        prediction_log_probs[i] = np.log(prob_y)

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
    exit()
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
