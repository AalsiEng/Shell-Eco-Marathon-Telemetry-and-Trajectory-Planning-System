import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib import cm
import matplotlib.cbook as cbook
import csv
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

x = []
y = []
Z = []

with open('For_Map.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        x.append(float(row[1]))
        y.append(float(row[2]))
        Z.append(int(float(row[3]) * 10 - 99))

fig, ax = plt.subplots(1, 1)
n = 60
z = np.array(Z)

colors = plt.cm.jet(np.linspace(0, 1, n))

plt.scatter(x, y, label='Lusail Coordinates', color = colors[z])

plt.title('Lusail Coordinates Plot')
plt.grid()
plt.legend()
plt.show()
 