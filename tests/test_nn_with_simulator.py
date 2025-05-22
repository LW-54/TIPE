import pytest
import numpy as np

from encephalon.serial_utils import simulate_serial
from encephalon.arduino_sim import make_arduino_simulator
from encephalon.nn import NN
from encephalon.types_and_functions import ReLU, Id

@pytest.fixture
def nn_with_sim():
    layers = [2, 4, 1]
    # simulator callback
    cb = make_arduino_simulator(layers, f=ReLU, noise_amplitude=0.0)
    interface, sim = simulate_serial(cb)
    sim.start()
    model = NN(layers, serial_interface=interface, f=ReLU, g=Id, verbose=False)
    yield model
    sim.stop()
    interface.close()

def test_nn_use_and_set(nn_with_sim):
    # Check that use returns a numeric array of correct shape
    x = np.array([0.1, -0.2])
    y = nn_with_sim.use(x)
    assert isinstance(y, np.ndarray)
    assert y.shape == (1, 1)

def test_nn_train_smoke(nn_with_sim):
    # tiny dataset: x->sum(x) regression
    data  = np.array([[1,2],[3,4],[5,6]])
    labels = data.sum(axis=1, keepdims=True)
    # one epoch, batch=2
    nn_with_sim.train(data, labels, epochs=1, batch_size=2, graphing=False)
    # After training, use still runs
    y2 = nn_with_sim.use(np.array([7,8]))
    assert y2.shape == (1,1)
