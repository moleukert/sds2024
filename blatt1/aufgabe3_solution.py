# Lösung von Moodle übung 3
import numpy as np


def load_honda_city_dataset():
    """
    Liest den Honda City Datensatz als NumPy Arrays ein. 

    Output:

    year_built = np.array[int],
    km_driven = np.array[int],
    selling_price = np.array[float]
    """
    X = np.loadtxt("honda_city.csv", delimiter=",", skiprows=1, dtype=float)

    year_built = np.asarray(X[:, 0], dtype=int)
    km_driven = np.asarray(X[:, 1], dtype=int)
    selling_price = X[:, 2]

    return year_built, km_driven, selling_price


def teilaufgabe_a():
    """
    Nutzen Sie den Datensatz, um eine multivariate lineare Regression zu implementieren, die mithilfe des Baujahres und des Kilometerstandes den Verkaufspreis schätzt. Implementieren Sie die algebraische Lösung zur Berechnung der Regression. Geben Sie die Schätzwerte für den Datensatz zurück.

    Output:

    y_pred = np.array[float]
    """
    year_built, km_driven, selling_price = load_honda_city_dataset()

    # Erstellung der Featurematrix X mit einer zusätzlichen Spalte mit Einsen für den Intercept
    intercept = np.ones((year_built.shape[0], 1))
    features = np.column_stack((intercept, year_built, km_driven))

    # Umwandlung von selling_price in einen Spaltenvektor
    y = selling_price.reshape(-1, 1)

    # Berechnung der Koeffizienten anhand der algebraischen Lösung
    theta = np.linalg.inv(features.T @ features) @ features.T @ y

    # Berechnung der Schätzungen
    y_pred = features @ theta

    # Schätzungen aus Konsistenzgründen als 1D-Array zurückgeben
    return y_pred.flatten()


def teilaufgabe_b():
    """
    Berechnen Sie den ''Root Mean Square Error'' der Schätzwerte (bezüglich der echten Verkaufspreise) aus Aufgabe 3 (a). Notieren Sie sitchhaltig, als Kommentar, was dieser Fehlerwert im Kontext dieser Aufgabe bedeutet.

    Output:

    rmse = float
    """
    year_built, km_driven, selling_price = load_honda_city_dataset()
    y_pred = teilaufgabe_a()

    # Berechnung des mittleren quadratischen Fehlers
    msr = np.mean((selling_price - y_pred) ** 2)

    # Berechnung der mittleren quadratischen Fehlerwurzel
    rmse = np.sqrt(msr)

    return rmse

    '''
    Was bedeutet der RMSE im Kontext dieser Aufgabe?
    Bedeutung: The RMSE in this context would quantify how accurate the predictions of the price using multilinear regression with the year built and km driven are.
               An error of about 0.94 would mean that the predictions are on average about 940$ off.  
               It is hard to evaluate how good the RMSE of about 0.94 is in this situation but being about 1k off is not optimal, still the prediction seems decent.

               Der RMSE quantifiziert die Prognosegenauigkeit. In diesem Kontext bezieht er sich konkret auf die Prognose des Verkaufspreises mithilfe von multivariater
               linearer Regression unter Berücksichtigung des Baujahres und des Kilometerstandes. Der RMSE wird in der Einheit der Zielvariablen angegeben, 
               daher bedeutet ein Fehler von circa 0.94, dass die Prognosen der Verkaufspreise im Durchschnitt um 940$ von den tatsächlichen Werten abweichen. 
               Die genaue Relevanz hängt vom Kontext ab, fast 1 Tsd. Dollar Abweichung ist definitiv nicht perfekt, aber für lineare Regression wirkt es akzeptabel.
    '''


if __name__ == "__main__":
    print(f"Teilaufgabe a:\n{teilaufgabe_a()}")
    print(f"Teilaufgabe b: {teilaufgabe_b():.3f}")
