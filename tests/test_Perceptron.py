"""
@auther: Tziporah
"""
import pytest

from perceptron import Perceptron


@pytest.fixture
def inputs():
    return [0.8, 0.9]


@pytest.fixture
def eta():
    return 5


@pytest.fixture
def desired_output():
    return 0.95


@pytest.fixture
def perceptron():
    init_weights = [0.24, 0.88]
    bias = 0

    return Perceptron(init_weights, bias)


def test_calc_activity(inputs, perceptron):
    perceptron.calc_activity(inputs, bias=perceptron.bias)

    assert perceptron.activity == 0.984


def test_calc_activation(inputs, perceptron):
    perceptron.calc_activation(inputs, bias=perceptron.bias)
    assert perceptron.activation == 0.7279011823597308


def test_set_delta_weights(inputs, desired_output, eta, perceptron):
    act = 0.7279011823597308
    delta = act * (1 - act) * (desired_output - act)
    perceptron.delta = delta
    perceptron.set_delta_weights(inputs, eta)
    print(perceptron.delta_weights)

    assert perceptron.delta_weights == [0.17595650106097208, 0.19795106369359358]


def test_update_weights(perceptron):
    perceptron.delta_weights = [0.17595650106097208, 0.19795106369359358]
    perceptron.update_weights()

    assert perceptron.weights == [0.41595650106097204, 1.0779510636935936]


def test_update_bias(eta, perceptron):
    perceptron.delta = 0.04398912526524302
    perceptron.update_bias(eta)

    assert perceptron.bias == 0.2199456263262151


def test_perceptron(perceptron, inputs, desired_output, eta):
    for i in range(75):
        perceptron.calc_activation(inputs, perceptron.bias)
        delta = perceptron.activation * (1 - perceptron.activation) * (desired_output - perceptron.activation)
        perceptron.delta = delta
        perceptron.set_delta_weights(inputs, eta)
        perceptron.update_weights()
        perceptron.update_bias(eta)

    assert perceptron.activation == 0.9474146590310214
