# -------------------------------
# Abgabegruppe:
# Personen:
# HU-Accountname:
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
    y_pred = None
    
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
    rmse = None
    
    return rmse

    '''
    Was bedeutet der RMSE im Kontext dieser Aufgabe?
    Bedeutung: 
    '''


if __name__ == "__main__":
    print(f"Teilaufgabe a:\n{teilaufgabe_a()}")
    print(f"Teilaufgabe b: {teilaufgabe_b()}")