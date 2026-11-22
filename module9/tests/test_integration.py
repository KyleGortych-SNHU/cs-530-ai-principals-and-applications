"""Integration tests for Module 9 workflows."""

import numpy as np

from module9 import neural_network, payroll


def test_xor_training_and_prediction_pipeline():
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
    record = payroll.calculate_payroll("EMP-42", 40, 30.0, "IT")

    assert record["employee_id"] == "EMP-42"
    assert record["taxable_income"] == record["gross_pay"] + record["bonus"]
    assert record["net_pay"] == record["taxable_income"] - record["taxes"]
