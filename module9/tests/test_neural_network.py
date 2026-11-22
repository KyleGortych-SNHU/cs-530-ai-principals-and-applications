"""Unit tests for the Module 9 neural network."""

from unittest.mock import patch

import numpy as np
import pytest

from module9 import neural_network


def test_sigmoid_returns_probabilities():
    """Specification: sigmoid returns finite probability values.

    Precondition: the input contains finite scalar values.
    Postcondition: output shape and probability bounds are unchanged/valid.
    Effect: no input or external state is mutated.
    """
    values = np.array([-1000.0, 0.0, 1000.0])

    result = neural_network.sigmoid(values)

    assert result.shape == values.shape
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert np.isclose(result[1], 0.5)


def test_sigmoid_rejects_non_finite_values():
    """Specification: sigmoid rejects non-finite input."""
    with pytest.raises(ValueError):
        neural_network.sigmoid(np.array([np.nan]))


def test_forward_rejects_wrong_input_columns():
    """Specification: forward rejects inputs with the wrong feature count."""
    params = np.zeros(21, dtype=np.float64)

    with pytest.raises(ValueError):
        neural_network.forward(params, np.zeros((1, 2), dtype=np.float64))


def test_backpropagation_matches_finite_difference():
    """Specification: analytical gradient matches a finite-difference gradient.

    Precondition: parameters and XOR data are finite.
    Postcondition: finite-difference and backpropagation gradients agree.
    Effect: no model state is mutated.
    """
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
    """Specification: repeated seeded training produces repeatable weights."""
    first = neural_network.train(seed=9, max_iterations=30)
    second = neural_network.train(seed=9, max_iterations=30)

    np.testing.assert_allclose(first.x, second.x)


@pytest.mark.parametrize(
    ("activation", "expected"),
    [("sigmoid", 0.8807970779778823), ("step", 1.0)],
)
def test_single_neuron_supports_both_activations(activation, expected):
    """Specification: a neuron supports exactly sigmoid and step outputs.

    Precondition: activation is sigmoid or step.
    Postcondition: sigmoid is a probability and step is binary.
    Effect: no input or external state is mutated.
    """
    result = neural_network.single_neuron([1.0, 1.0], [1.0, 1.0], activation)

    assert np.isclose(result, expected)


def test_single_neuron_rejects_unknown_activation():
    """Specification: unsupported neuron activation names are rejected."""
    with pytest.raises(ValueError, match="activation"):
        neural_network.single_neuron([1.0], [1.0], "unknown")


def test_synthetic_data_set_is_reproducible_and_valid():
    """Specification: synthetic data is reproducible and binary.

    Precondition: default sample count and seed are used.
    Postcondition: features and labels are repeatable and labels are 0 or 1.
    Effect: no external state is mutated.
    """
    first = neural_network.create_synthetic_data_set()
    second = neural_network.create_synthetic_data_set()

    assert first[0].shape == (200, 2)
    assert first[1].shape == (200,)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(first[1]) <= {0.0, 1.0}


@pytest.mark.parametrize("activation", ["sigmoid", "step"])
def test_calculate_weights_and_binary_classifier(activation):
    """Specification: each activation calculates weights and predictions.

    Precondition: activation is sigmoid or step for a linearly separable set.
    Postcondition: weights have bias plus feature size and predictions are
    valid binary values with at least 95 percent accuracy.
    Effect: no external state is mutated.
    """
    features, targets = neural_network.create_synthetic_data_set(
        sample_count=300,
        seed=17,
    )
    weights = neural_network.calculate_weights(
        (features, targets),
        activation=activation,
        learning_rate=0.1,
        max_iterations=500,
    )
    predictions = neural_network.binary_classifier(
        features,
        weights,
        activation=activation,
    )

    assert weights.shape == (3,)
    assert predictions.shape == (300,)
    assert set(predictions) <= {0.0, 1.0}
    assert np.mean(predictions == targets) >= 0.95


def test_binary_classifier_calls_single_neuron_for_each_row():
    """Specification: classifier delegates each row to the single-neuron API.

    Precondition: each dataset row is a valid feature vector and weights
    contain one bias and one weight per feature.
    Postcondition: single_neuron is called once per row and predictions are
    binary.
    Effect: only the test's spy observes neuron calls.
    """
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = np.array([0.0, 1.0, -1.0])

    with patch.object(
        neural_network,
        "single_neuron",
        wraps=neural_network.single_neuron,
    ) as neuron_spy:
        predictions = neural_network.binary_classifier(
            features,
            weights,
            activation="step",
        )

    assert neuron_spy.call_count == len(features)
    assert set(predictions) <= {0.0, 1.0}
