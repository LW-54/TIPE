import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from encephalon.grapher import Grapher
from encephalon.types_and_functions import ReLU, Id, tanh, sigmoid
from encephalon.serial_utils import SerialJSONInterface, handshake

interface = SerialJSONInterface(port="/dev/ttyACM0", baud=9600, timeout=1.0)
handshake(interface)

gr = Grapher(interface,verbose=True)

gr.set(1,1)
print(gr.read(1))

x_min,x_max,n = 0, 5, 25

X = np.linspace(x_min,x_max,n).reshape(-1,1)  # shape: (n, 1)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(X.flatten(),gr.graph(X).flatten()) # flatten to (n,)

f = tanh[0]

ax.plot(X.flatten(),f(X).flatten())

plt.show()