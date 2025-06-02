import numpy as np

from encephalon.nn import NN
from encephalon.types_and_functions import ReLU, Id, tanh, sigmoid
from encephalon.serial_utils import simulate_serial, SerialJSONInterface, handshake
from encephalon.arduino_sim import make_arduino_simulator
from encephalon.representations import auto_subplot, graph_3d, decision_boundary


data = [[1,1], [0,1], [1,0], [0,0]]
labels = [[0], [1], [1], [0]]


layers, f, g = [2, 3, 1], tanh, Id

if True :
    interface, sim = simulate_serial(make_arduino_simulator(layers, f=f, noise_amplitude=0))
    sim.start()
else :
    interface = SerialJSONInterface(port="/dev/ttyACM0", baud=9600, timeout=1.0)

handshake(interface)

xor = NN(interface, layers, name="xor", f=f, g=g, verbose=True)


xor.train(data, labels, epochs=1000, batch_size=1, graphing=True)


print(xor.use([1,1]))
print(xor.use([0,1]))
print(xor.use([1,0]))
print(xor.use([0,0]))

# print(xor.W)
# print(xor.b)


to_plot = [
    (graph_3d, dict(model=xor, x_min=0, x_max=1, y_min=0, y_max=1, n=20,)),
    (decision_boundary, dict(model=xor, x_min=0, x_max=1, y_min=0, y_max=1, n=50, boundary=0.5, data_0=[[1,1], [0,0]], data_1=[[0,1], [1,0]])),
]

auto_subplot(1, 2, to_plot, figsize=(10, 5))



sim.stop()
interface.close()

#function in interval
#is inside a triangle
#circles
#iris
#breast cancer
#other logical functions
#maybe something more scientific (chemistry) conductivity
#make code to plot components
