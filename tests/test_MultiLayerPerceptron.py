"""
@auther: Tziporah
"""
import pytest

from MultiLayerPerceptron import MLP


@pytest.fixture
def init_inputs():
    return [1, 2]


@pytest.fixture
def init_weights():
    return [[0.3, 0.3], [0.3, 0.3], [0.8, 0.8]]


@pytest.fixture
def biases():
    return [None] * 3


@pytest.fixture
def desired_output():
    return 0.7


@pytest.fixture
def eta():
    return 1


def test_ffbp(init_inputs, init_weights, biases, desired_output, eta):
    num_nodes = 2
    mlp = MLP(n_hidden=num_nodes, weights=init_weights, biases=biases)

    mlp.ffbp(inputs=init_inputs, desired_output=desired_output, eta=eta)
    mlp._feed_forward(init_inputs, desired_output=desired_output)
    assert mlp.output_layers[0].activation == 0.7547415782405359


def test__feed_forward(init_inputs, init_weights, biases, desired_output):
    num_nodes = 2
    mlp = MLP(n_hidden=num_nodes, weights=init_weights, biases=biases)

    ff_output = mlp._feed_forward(init_inputs, desired_output=desired_output)
    assert ff_output == [0.7109495026250039, 0.7109495026250039]


def test__back_propagate(init_inputs, init_weights, biases, eta):
    num_nodes = 2
    mlp = MLP(n_hidden=num_nodes, weights=init_weights, biases=biases)
    ff_output = [0.7109495026250039, 0.7109495026250039]
    mlp.delta_o = [-0.01051980066099822]

    mlp._back_propagate(init_inputs, ff_output, eta)

    updated_weights_h = [i.weights for i in mlp.hidden_layers]
    assert updated_weights_h == [[0.2982705421847884, 0.2965410843695768], [0.2982705421847884, 0.2965410843695768]]

