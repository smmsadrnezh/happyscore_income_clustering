from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import csv
import numpy as np
from tkinter import *


def plot_clusters(income, happiness):
    # Find Centroids
    k = 3
    data = np.array(list(zip(income, happiness)))
    kmeans_model = KMeans(n_clusters=k).fit(data)
    centers = np.array(kmeans_model.cluster_centers_)

    # Plot Clusters
    plt.plot()

    plt.title('Happy Score by Income for Countries\nK-Means Centroids')
    plt.xlabel('Income')
    plt.ylabel('Happiness')

    colors = ['b', 'g', 'r']
    for i, l in enumerate(kmeans_model.labels_):
        plt.scatter(income[i], happiness[i], color=colors[l])

    plt.scatter(centers[:, 0], centers[:, 1], color='black', s=400, linewidths=1000, edgecolors='red', alpha=0.75)
    plt.show()


def plot_data(income, happiness):
    plt.title('Happy Score by Income for Countries')
    plt.xlabel('Income')
    plt.ylabel('Happiness')

    plt.scatter(income, happiness)
    plt.show()


def gui(income, happiness):
    window = Tk()
    window.title("Happiness/Income Clustering")
    window.geometry('110x60')

    lbl_plot_data = Label(window, text="Data: ")
    lbl_plot_cluster = Label(window, text="Clusters: ")
    btn_plot_data = Button(window, text="Plot", command=lambda: plot_data(income, happiness))
    btn_plot_cluster = Button(window, text="Plot", command=lambda: plot_clusters(income, happiness))

    lbl_plot_data.grid(column=0, row=0)
    lbl_plot_cluster.grid(column=0, row=1)
    btn_plot_data.grid(column=1, row=0)
    btn_plot_cluster.grid(column=1, row=1)

    window.mainloop()


def export_data(file_name):
    with open(file_name) as csv_file:
        countries = list(csv.DictReader(csv_file))
        income = [float(country['avg_income']) for country in countries]
        happiness = [float(country['happyScore']) for country in countries]
        return income, happiness


def main():
    income, happiness = export_data('happyscore_income.csv')
    gui(income, happiness)


if __name__ == '__main__':
    main()
