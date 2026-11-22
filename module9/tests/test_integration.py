"""Integration tests for Module 9 workflows."""

import numpy as np

from module9 import neural_network, payroll


def test_xor_training_and_prediction_pipeline():
    """Specification: XOR training trains and predicts four rows."""
    result = neural_network.train(seed=7, max_iterations=300)
    predictions = neural_network.forward(result.x, neural_network.XOR_INPUTS)
    loss = neural_network.mean_squared_error(
        result.x,
        neural_network.XOR_INPUTS,
        neural_network.XOR_TARGETS,
    )

    assert np.isfinite(result.x).all()
    assert loss < 0.5
    assert predictions.shape == (4, 1)


def test_payroll_record_is_consistent_across_calculation():
    """Specification: integration payroll totals remain consistent."""
    record = payroll.calculate_payroll("EMP-42", 40, 30.0, "IT")

    assert record["employee_id"] == "EMP-42"
    assert record["taxable_income"] == record["gross_pay"] + record["bonus"]
    assert record["net_pay"] == record["taxable_income"] - record["taxes"]


def test_synthetic_classifier_pipeline_runs_both_activations():
    """Specification: generate data, calculate weights, and classify for both
    sigmoid and step activations.

    Precondition: synthetic features and targets are linearly separable.
    Postcondition: both paths return binary-predictable weight and prediction
    shapes.
    Effect: no external state is mutated.
    """
    data_set = neural_network.create_synthetic_data_set(
        sample_count=400,
        seed=23,
    )

    for activation in ("sigmoid", "step"):
        weights = neural_network.calculate_weights(
            data_set,
            activation=activation,
            learning_rate=0.1,
            max_iterations=500,
        )
        predictions = neural_network.binary_classifier(
            data_set,
            weights,
            activation=activation,
        )

        assert weights.shape == (3,)
        assert predictions.shape == (400,)
        assert set(predictions) <= {0.0, 1.0}
