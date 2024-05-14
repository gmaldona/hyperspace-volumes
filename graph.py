import matplotlib.pyplot as plt
import numpy as np
import csv

def main():
    dimensional_samples = []
    dimensions = np.arange(2, 17, 1)
    distances  = np.arange(0, 1, 0.01)

    with open('data.csv', newline='\n') as fp:
        rd = csv.reader(fp, delimiter=',')
        for row in rd:
            dimensional_samples.append(row)

    for i, _ in enumerate(dimensional_samples):
        for j, _ in enumerate(dimensional_samples[i]):
            dimensional_samples[i][j] = float(dimensional_samples[i][j])

    dimensional_samples = np.array(dimensional_samples)


    X, Y = np.meshgrid(distances, dimensions)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, dimensional_samples)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if __name__ == '__main__':
    main()