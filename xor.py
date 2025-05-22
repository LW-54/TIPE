import numpy as np

from encephalon.nn import NN
from encephalon.types_and_functions import ReLU, Id
from encephalon.serial_utils import simulate_serial
from encephalon.arduino_sim import make_arduino_simulator


data = [[1,1], [0,1], [1,0], [0,0]]
labels = [[0], [1], [1], [0]]


layers, f, g = [2, 3, 1], ReLU, Id

interface, sim = simulate_serial(make_arduino_simulator(layers, f=ReLU, noise_amplitude=0.0))
sim.start()

xor = NN(interface, layers, name="xor", f=f, g=g, verbose=True)

xor.train(data, labels, epochs=1000, batch_size=2, graphing=True)

print(xor.use([1,1]))
print(xor.use([0,1]))
print(xor.use([1,0]))
print(xor.use([0,0]))

print(xor.W)
print(xor.b)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = xor.use([x,y])
        points.append([x, y, z[0][0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()



sim.stop()
interface.close()