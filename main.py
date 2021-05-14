# Kevin Shah
# May 14, 2021

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
lm = LinearRegression()


def part_one():
    df = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])

    lm.fit(df, boston.target)

    chart = pd.DataFrame(zip(df.columns, lm.coef_))
    chart.columns = ["feature_names", "coef"]
    chart = chart.sort_values("coef", ascending=False)

    print(chart)
    print("Most positive influence = RM with coefficient of 3.809865")
    print("Most negative influence = NOX with coefficient of -17.766611")


import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import load_wine

wine = load_wine()


def part_two():
    df = pd.DataFrame(data=wine['data'], columns=wine['feature_names'])

    count = {}
    for i in range(1, 15):
        sol = KMeans(n_clusters=i).fit(df)
        count[i] = sol.inertia_

    x = list(count.keys())
    y = list(count.values())

    plt.scatter(x, y)
    plt.plot(x, y)
    plt.show()


part_one()
part_two()
