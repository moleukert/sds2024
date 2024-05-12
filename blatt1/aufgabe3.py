# -------------------------------
# Abgabegruppe: Gruppe 10
# Personen: Alisha Vaders, Moritz Leukert, Yann-Cédric Gagern
# HU-Accountname: leukertm, vadersal, gagernya
# -------------------------------
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

    year_built = np.asarray(X[:,0], dtype=int)
    km_driven = np.asarray(X[:,1], dtype=int)
    selling_price = X[:,2]

    return year_built, km_driven, selling_price


def teilaufgabe_a():
    """
    Nutzen Sie den Datensatz, um eine multivariate lineare Regression zu implementieren, die mithilfe des Baujahres und des Kilometerstandes den Verkaufspreis schätzt. Implementieren Sie die algebraische Lösung zur Berechnung der Regression. Geben Sie die Schätzwerte für den Datensatz zurück.

    Output:

    y_pred = np.array[float]
    """
    year_built, km_driven, selling_price = load_honda_city_dataset()

    # Implementieren Sie hier Ihre Lösung

    n = len(year_built)
    # initialize A with column of 1's, year_built, km_driven
    matrix_A = np.concatenate([np.full((n,), 1)[:,np.newaxis],year_built[:,np.newaxis],km_driven[:,np.newaxis]], axis=1)

    # check if operations can be performed
    if np.linalg.det(matrix_A.T @ matrix_A) == 0:
        print("Cannot invert A.T*A !")
        return None
    elif n < 2:
        print("Not enough samples, at least 2 needed!")
        return None

    # calculate k = (A.T*A)^-1 * A.T * y
    k = np.linalg.inv(matrix_A.T @ matrix_A) @ matrix_A.T @ selling_price[:,np.newaxis]
    # use k to calculate prediction
    y_pred = k[0] + k[1]*year_built + k[2]*km_driven

    return y_pred


def teilaufgabe_b():
    """
    Berechnen Sie den ''Root Mean Square Error'' der Schätzwerte (bezüglich der echten Verkaufspreise) aus Aufgabe 3 (a). Notieren Sie sitchhaltig, als Kommentar, was dieser Fehlerwert im Kontext dieser Aufgabe bedeutet.

    Output:

    rmse = float
    """
    year_built, km_driven, selling_price = load_honda_city_dataset()
    y_pred = teilaufgabe_a()

    # Implementieren Sie hier Ihre Lösung
    n = len(selling_price)
    # calculate rmse = (1/n sgrt( sum( (y-f(x))^2) ) )
    rmse = 1/n*np.sqrt(np.sum(np.square(selling_price-y_pred)))

    return rmse

    '''
    Was bedeutet der RMSE im Kontext dieser Aufgabe?
    Bedeutung: The RMSE in this context would quantify how accurate the predictions of the price using multilinear regression with the year built and km driven are.
               An error of about 0.184 would mean that the predictions are on average about 184$ off.  
               It is hard to evaluate how good the RMSE of about 0.184 is in this situation but given that it is decently close to 0, the prediction seems good.
    '''


if __name__ == "__main__":
    print(f"Teilaufgabe a:\n{teilaufgabe_a()}")
    print(f"Teilaufgabe b: {teilaufgabe_b()}")
