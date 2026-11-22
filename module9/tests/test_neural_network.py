"""Unit tests for the Module 9 neural network."""

import numpy as np
import pytest

from module9 import neural_network


def test_sigmoid_returns_probabilities():
    values = np.array([-1000.0, 0.0, 1000.0])

    result = neural_network.sigmoid(values)

    assert result.shape == values.shape
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert np.isclose(result[1], 0.5)


def test_sigmoid_rejects_non_finite_values():
    with pytest.raises(ValueError):
        neural_network.sigmoid(np.array([np.nan]))


def test_forward_rejects_wrong_input_columns():
    params = np.zeros(21, dtype=np.float64)

    with pytest.raises(ValueError):
        neural_network.forward(params, np.zeros((1, 2), dtype=np.float64))


def test_backpropagation_matches_finite_difference():
    params = np.random.default_rng(1).normal(size=21)
    inputs = neural_network.XOR_INPUTS
    targets = neural_network.XOR_TARGETS
    epsilon = 1e-6
    expected = np.empty_like(params)

    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        expected[index] = (
            neural_network.mean_squared_error(plus, inputs, targets)
            - neural_network.mean_squared_error(minus, inputs, targets)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        neural_network.gradient(params, inputs, targets),
        expected,
        atol=1e-6,
    )


def test_training_is_repeatable_for_same_seed():
    first = neural_network.train(seed=9, max_iterations=30)
    second = neural_network.train(seed=9, max_iterations=30)

    np.testing.assert_allclose(first.x, second.x)
