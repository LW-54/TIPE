import numpy as np

from encephalon.nn import NN
from encephalon.types_and_functions import ReLU, Id, tanh
from encephalon.serial_utils import simulate_serial
from encephalon.arduino_sim import make_arduino_simulator
from encephalon.representations import auto_subplot, graph_2d

dim = 1

x_min, x_max= -np.pi*4, np.pi*4

data = np.random.uniform(x_min, x_max, (64*10**2, dim))
labels = np.sin(data)


name = "foo"

layers, f, g = [dim, 16*dim, 16*dim, dim], tanh, tanh

interface, sim = simulate_serial(make_arduino_simulator(layers, f=f, noise_amplitude=0.0))
sim.start()

func = NN(interface, layers, name=name, f=f, g=g, verbose=True)


func.train(data, labels, epochs=1000, batch_size=64, graphing=False)


to_plot = [
    (graph_2d, dict(model=func, x_min=x_min, x_max=x_max, n=100,)),
]

auto_subplot(1, 1, to_plot, figsize=None)



sim.stop()
interface.close()