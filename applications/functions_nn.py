import numpy as np

from encephalon.nn import NN
from encephalon.types_and_functions import Id, tanh
from encephalon.serial_utils import simulate_serial, SerialJSONInterface, handshake
from encephalon.arduino_sim import make_arduino_simulator
from encephalon.representations import auto_subplot, graph_2d


func = np.sin

x_min,x_max,n = 0, np.pi, 25 #

data = np.linspace(x_min,x_max,n).reshape(-1,1)  # shape: (n, 1)

labels = func(data)


name = "foo"

layers, f, g = [1,3,1], tanh, Id

if False :
    interface, sim = simulate_serial(make_arduino_simulator(layers, f=f, noise_amplitude=0))
    sim.start()
else :
    interface = SerialJSONInterface(port="/dev/ttyACM0")

handshake(interface)


model = NN(interface, layers, name=name, f=f, g=g, verbose=True)


model.train(data, labels, epochs=1000, batch_size=1, graphing=False)

print(model.W)
print(model.b)

to_plot = [
    (graph_2d, dict(model=model, x_min=x_min, x_max=x_max, n=25, func=func,)),
]

auto_subplot(1, 1, to_plot, figsize=None)


interface.close()
sim.stop()