import pytest
import numpy as np
from encephalon.arduino_sim import make_arduino_simulator
from encephalon.types_and_functions import ReLU, Id

@pytest.fixture
def sim_callback():
    # 3‐layer net: input=2, hidden=3, output=1
    return make_arduino_simulator([2,3,1], f=ReLU, noise_amplitude=0.0)

def test_set_weights_and_biases_and_forward(sim_callback):
    # send new W & b and then a forward
    # W1: shape (2,3), W2: (3,1)
    W = {"1": [[1,2,3],[4,5,6]], "2": [[7],[8],[9]]}
    b = {"1": [0,0,0], "2": [0]}
    msg = {"cmd":"set_weights_and_biases","W":W,"b":b}
    resp = sim_callback(msg)
    assert resp["status"] == "OK"

    # forward on one example [1,1]
    fout = sim_callback({"cmd":"forward","input":[1,1]})
    # manual: hidden = ReLU([1,1]@W1 + b1) = ReLU([1+4,2+5,3+6]) = [5,7,9]
    # output = [5,7,9]@W2 + b2 = 5*7 +7*8 +9*9 = 35+56+81=172
    assert pytest.approx(172) == fout["output"][0][0]

def test_set_and_read_commands(sim_callback):
    # set a single weight W1[0,1]=42
    r1 = sim_callback({"cmd":"set","param":"W1:0:1","value":42})
    assert r1["status"] == "OK"
    # set bias b2[0]=5
    r2 = sim_callback({"cmd":"set","param":"b2:0","value":5})
    assert r2["status"] == "OK"
    # set input pin 1
    r3 = sim_callback({"cmd":"set","param":"I1","value":2.5})
    assert r3["status"] == "OK"
    # read output 0
    r4 = sim_callback({"cmd":"read","index":0})
    assert "value" in r4 and isinstance(r4["value"], float)

def test_set_and_read_errors(sim_callback):
    # bad param string
    bad = sim_callback({"cmd":"set","param":"X23","value":1})
    assert "error" in bad

    # out-of-range read
    err = sim_callback({"cmd":"read","index":999})
    assert "error" in err

    # unknown command
    unk = sim_callback({"cmd":"foobar"})
    assert "error" in unk
