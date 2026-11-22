"""Train a small neural network to solve an XOR-style data set.

Date: 11/22/2026
Author: Kyle Gortych
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

FloatArray = NDArray[np.float64]

INPUT_SIZE: Final = 3
HIDDEN_SIZE: Final = 4
OUTPUT_SIZE: Final = 1
XOR_INPUTS: Final = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    dtype=np.float64,
)
XOR_TARGETS: Final = np.array(
    [[0.0], [1.0], [1.0], [0.0]],
    dtype=np.float64,
)


def sigmoid(values: FloatArray) -> FloatArray:
    """Apply sigmoid safely to an array.

    Precondition: values contains finite numeric values.
    Postcondition: every returned value is in the closed interval [0, 1].
    Effect: no state is mutated.
    """
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite numbers")
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))


def _forward_components(
    parameters: FloatArray, inputs: FloatArray
) -> tuple[FloatArray, FloatArray]:
    parameter_count = (
        INPUT_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + HIDDEN_SIZE * OUTPUT_SIZE
        + OUTPUT_SIZE
    )
    if parameters.shape != (parameter_count,) or inputs.ndim != 2:
        raise ValueError("invalid parameter or input shape")
    if inputs.shape[1] != INPUT_SIZE:
        raise ValueError(f"inputs must contain exactly {INPUT_SIZE} columns")
    if not np.isfinite(parameters).all() or not np.isfinite(inputs).all():
        raise ValueError(
            "parameters and inputs must contain only finite numbers"
        )

    first_weights_end = INPUT_SIZE * HIDDEN_SIZE
    first_biases_end = first_weights_end + HIDDEN_SIZE
    second_weights_end = first_biases_end + HIDDEN_SIZE * OUTPUT_SIZE
    weights_input_hidden = parameters[:first_weights_end].reshape(
        INPUT_SIZE, HIDDEN_SIZE
    )
    biases_hidden = parameters[first_weights_end:first_biases_end]
    weights_hidden_output = parameters[
        first_biases_end:second_weights_end
    ].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    biases_output = parameters[second_weights_end:]

    hidden = sigmoid(inputs @ weights_input_hidden + biases_hidden)
    predictions = sigmoid(hidden @ weights_hidden_output + biases_output)
    return hidden, predictions


def forward(parameters: FloatArray, inputs: FloatArray) -> FloatArray:
    """Calculate network predictions.

    Precondition: parameters has the expected size; inputs is two-dimensional
    with INPUT_SIZE columns.
    Postcondition: returns one prediction per input in a two-dimensional array.
    Invariant: all weights and biases have a one-to-one mapping to the network.
    Effect: no state or input data is mutated.
    """
    return _forward_components(parameters, inputs)[1]


def mean_squared_error(
    parameters: FloatArray,
    inputs: FloatArray,
    targets: FloatArray,
) -> float:
    """Calculate mean squared prediction error.

    Precondition: inputs and targets have matching, valid shapes.
    Postcondition: a finite, non-negative scalar is returned.
    Effect: no state is mutated.
    """
    if targets.shape != (inputs.shape[0], OUTPUT_SIZE):
        raise ValueError("targets must have one value per input")
    if not np.isfinite(targets).all():
        raise ValueError("targets must contain only finite numbers")
    predictions = forward(parameters, inputs)
    return float(np.mean(np.square(predictions - targets)))


def gradient(
    parameters: FloatArray,
    inputs: FloatArray,
    targets: FloatArray,
) -> FloatArray:
    """Calculate the loss gradient using vectorized backpropagation.

    Precondition: parameters is a valid flat parameter vector.
    Postcondition: a gradient with the same shape as parameters is returned.
    Effect: no state or input data is mutated.
    """
    if (
        targets.shape != (inputs.shape[0], OUTPUT_SIZE)
        or not np.isfinite(targets).all()
    ):
        raise ValueError("targets must be finite and have one value per input")

    hidden, predictions = _forward_components(parameters, inputs)
    count = targets.size
    output_delta = (
        2.0
        * (predictions - targets)
        * predictions
        * (1.0 - predictions)
        / count
    )

    first_weights_end = INPUT_SIZE * HIDDEN_SIZE
    first_biases_end = first_weights_end + HIDDEN_SIZE
    second_weights_end = first_biases_end + HIDDEN_SIZE * OUTPUT_SIZE
    weights_hidden_output = parameters[
        first_biases_end:second_weights_end
    ].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    gradient_weights_hidden_output = hidden.T @ output_delta
    gradient_biases_output = output_delta.sum(axis=0)
    hidden_delta = (
        (output_delta @ weights_hidden_output.T) * hidden * (1.0 - hidden)
    )
    gradient_weights_input_hidden = inputs.T @ hidden_delta
    gradient_biases_hidden = hidden_delta.sum(axis=0)
    return np.concatenate(
        [
            gradient_weights_input_hidden.ravel(),
            gradient_biases_hidden,
            gradient_weights_hidden_output.ravel(),
            gradient_biases_output,
        ]
    )


def train(
    inputs: FloatArray = XOR_INPUTS,
    targets: FloatArray = XOR_TARGETS,
    seed: int = 42,
    max_iterations: int = 1000,
) -> optimize.OptimizeResult:
    """Train network parameters with bounded-memory BFGS.

    Precondition: inputs and targets are finite and compatible; seed is
    non-negative and max_iterations is positive.
    Postcondition: result.x has the required parameter shape.
    Invariant: optimizer parameters always use the same forward-network layout.
    Effect: NumPy's RNG is initialized locally; no external state is mutated.
    """
    if inputs.ndim != 2 or inputs.shape[1] != INPUT_SIZE:
        raise ValueError(
            f"inputs must be a two-dimensional array with {INPUT_SIZE} columns"
        )
    if targets.shape != (inputs.shape[0], OUTPUT_SIZE):
        raise ValueError("targets must have one value per input")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("inputs and targets must contain only finite numbers")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    rng = np.random.default_rng(seed)
    weights_input_hidden = rng.normal(size=INPUT_SIZE * HIDDEN_SIZE)
    weights_hidden_output = rng.normal(size=HIDDEN_SIZE * OUTPUT_SIZE)
    biases_hidden = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    biases_output = np.zeros(OUTPUT_SIZE, dtype=np.float64)
    initial_parameters = np.concatenate(
        [
            weights_input_hidden,
            biases_hidden,
            weights_hidden_output,
            biases_output,
        ]
    )

    result = optimize.minimize(
        mean_squared_error,
        initial_parameters,
        args=(inputs, targets),
        method="L-BFGS-B",
        jac=gradient,
        options={"maxiter": max_iterations},
    )
    if result.x.shape != initial_parameters.shape:
        raise RuntimeError("optimizer returned an invalid parameter vector")
    return result


if __name__ == "__main__":
    training_result = train()
    predictions = forward(training_result.x, XOR_INPUTS)
    print(
        "Loss:", mean_squared_error(training_result.x, XOR_INPUTS, XOR_TARGETS)
    )
    print("Predictions:")
    print(predictions)
