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
# #    0,1,2,3,4,5,6,0,1,2,3,4,5,6
# #    0,1,2,3,4,5,6,7,8,9,10,11,12,13
# M = [0,1,2,3,4,5,0,1,2,3,4,0,0,1]
# for i in range(len(M)):
#     gr.set(i,M[i])
#     print(gr.read(0))


x_min,x_max,n = 0, 5, 25

X = np.linspace(x_min,x_max,n).reshape(-1,1)  # shape: (n, 1)

fig = plt.figure()
ax = fig.add_subplot()

def g(X: np.ndarray, dac : int = 0, out : int = 0) -> np.ndarray:
    X = np.array(X, ndmin=2)
    Y = []
    n,_ = X.shape
    for i in range(n):
        gr.set(dac,X[i][0])
        Y.append(gr.read(out))
    return np.array(Y, ndmin=2) # dim not realy necessary


ax.plot(
    X.flatten(),
    g(X,int(input("dac to test : "))).flatten(),
    label="dac",
)


ax.grid(True)

# Add title and axis labels

ax.set_xlabel("Tension en V")


# Add the legend
ax.legend()

plt.show()
interface.close()