import numpy as np


def test():
    n = 5
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([9.50, 21.86, 30.65, 45.52, 49.77])
    # co-variance
    s_xy = 1 / (n - 1) * np.sum((x - np.mean(x)) * (y - np.mean(y)))
    # standard deviations
    s_x = np.sqrt(1 / (n - 1) * np.sum(np.square(x - np.mean(x))))
    s_y = np.sqrt(1 / (n - 1) * np.sum(np.square(y - np.mean(y))))
    print(f"Co-variance = {s_xy:.2f}")
    print(f"Co-variance np = {np.cov(x, y, ddof=1)[0, 1]:.2f}")  # numpy based automated co-variance ddof=1 -> 1/n-1
    # pearson correlation (needs to be between -1 and 1)
    r_xy = s_xy / (s_x * s_y)
    print(f"Pearson correlation = {r_xy:.2f}")
    print(f"Person correlation np = {np.corrcoef(x, y)[0, 1]:.2f}")
    # f(x) = kx+d
    k = r_xy * s_y / s_x
    d = np.mean(y) - k * np.mean(x)
    y_3 = k * x[2] + d
    # f(3)
    print(f"k = {k:.2f}")
    print(f"d = {d:.2f}")
    print(f"y_3 = {y_3}")


if __name__ == "__main__":
    test()
    pass
