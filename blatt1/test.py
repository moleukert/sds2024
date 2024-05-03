import numpy as np


def test():
    n = 5
    x = np.asarray([1,2,3,4,5],dtype=int)
    y = np.asarray([9.5,21.86,30.65,45.52,49.77],dtype=float)
    # co-variance
    s_xy = 1/(n-1) * np.sum((x-np.mean(x))*(y-np.mean(y)))
    # standard deviations
    s_x = np.sqrt(1/n* np.sum(np.square(x-np.mean(x))))
    s_y = np.sqrt(1/n* np.sum(np.square(y-np.mean(y))))
    print(s_xy)
    # pearson correlation (above 1?)
    r_xy = s_xy/(s_x*s_y)
    print(r_xy)
    # f(x) = kx+d
    k = r_xy * s_y/s_x
    d = np.mean(y)- k * np.mean(x)
    y_3 = k*x[2]+d
    # f(3)
    print(y_3)

if __name__ == "__main__":
    test()
    pass