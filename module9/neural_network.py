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
SUPPORTED_ACTIVATIONS: Final = ("sigmoid", "step")

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


def _validate_activation(activation: str) -> str:
    """Normalize and validate a supported activation name."""
    if not isinstance(activation, str):
        raise TypeError("activation must be a string")
    normalized = activation.strip().lower()
    if normalized not in SUPPORTED_ACTIVATIONS:
        choices = ", ".join(SUPPORTED_ACTIVATIONS)
        raise ValueError(f"activation must be one of: {choices}")
    return normalized


def single_neuron(
    inputs: FloatArray | list[float],
    weights: FloatArray | list[float],
    activation: str = "sigmoid",
) -> float:
    """Calculate one neuron output for a feature vector.

    Precondition: inputs and weights are finite one-dimensional vectors with
    the same length.
    Postcondition: returns a scalar in [0, 1] for sigmoid and a binary scalar
    of 0 or 1 for step activation.
    Effect: no input data or external state is mutated.
    """
    selected_activation = _validate_activation(activation)
    input_array = np.asarray(inputs, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if input_array.ndim != 1 or weight_array.ndim != 1:
        raise ValueError("inputs and weights must be one-dimensional")
    if input_array.size != weight_array.size:
        raise ValueError("inputs and weights must have the same length")
    if (
        not np.isfinite(input_array).all()
        or not np.isfinite(weight_array).all()
    ):
        raise ValueError("inputs and weights must contain only finite numbers")

    weighted_sum = float(np.dot(input_array, weight_array))
    if selected_activation == "sigmoid":
        return float(1.0 / (1.0 + np.exp(-np.clip(weighted_sum, -500, 500))))
    return 1.0 if weighted_sum >= 0.0 else 0.0


def create_synthetic_data_set(
    sample_count: int = 200,
    seed: int = 42,
) -> tuple[FloatArray, FloatArray]:
    """Create a reproducible linearly separable binary-classification data set.

    Precondition: sample_count is positive and seed is non-negative.
    Postcondition: returns finite feature and target arrays with matching
    row counts. Targets contain only 0 and 1.
    Effect: no external state is mutated.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    features = rng.uniform(-5.0, 5.0, size=(sample_count, 2))
    labels = (features[:, 0] + features[:, 1] >= 0.0).astype(np.float64)
    return features.astype(np.float64), labels


def _extract_data_set(
    data_set: object,
) -> tuple[FloatArray, FloatArray | None]:
    """Extract features and optional targets from common data-set formats."""
    if isinstance(data_set, tuple):
        if len(data_set) not in (1, 2):
            raise ValueError(
                "data_set tuple must contain features or features and targets"
            )
        features = np.asarray(data_set[0], dtype=np.float64)
        targets = (
            np.asarray(data_set[1], dtype=np.float64).reshape(-1)
            if len(data_set) == 2
            else None
        )
    elif isinstance(data_set, dict):
        if "features" in data_set:
            features = np.asarray(data_set["features"], dtype=np.float64)
        elif "data" in data_set:
            features = np.asarray(data_set["data"], dtype=np.float64)
        elif "x" in data_set:
            features = np.asarray(data_set["x"], dtype=np.float64)
        else:
            raise ValueError("data_set dictionary must contain features")
        target_key = next(
            (key for key in ("targets", "labels", "y") if key in data_set),
            None,
        )
        targets = (
            np.asarray(data_set[target_key], dtype=np.float64).reshape(-1)
            if target_key is not None
            else None
        )
    else:
        array = np.asarray(data_set, dtype=np.float64)
        if array.ndim == 1:
            features = array.reshape(-1, 1)
            targets = None
        else:
            features = array
            targets = None

    if features.ndim != 2:
        raise ValueError("feature data must be two-dimensional")
    if not np.isfinite(features).all():
        raise ValueError("feature data must contain only finite numbers")
    if targets is not None:
        if targets.shape != (features.shape[0],):
            raise ValueError("targets must contain one value per feature row")
        if not np.isfinite(targets).all():
            raise ValueError("targets must contain only finite numbers")
    return features, targets


def calculate_weights(
    data_set: object,
    activation: str = "sigmoid",
    learning_rate: float = 0.05,
    max_iterations: int = 500,
) -> FloatArray:
    """Calculate weights for a one-layer binary classifier.

    Precondition: data_set contains features and binary targets; activation
    is sigmoid or step; learning rate is positive and iterations is positive.
    Postcondition: returns a flat weight vector with one bias and one weight
    per feature.
    Effect: no external state or input arrays are mutated.
    """
    selected_activation = _validate_activation(activation)
    features, targets = _extract_data_set(data_set)
    if targets is None:
        raise ValueError("data_set must include targets")
    if not np.all(np.isin(targets, (0.0, 1.0))):
        raise ValueError("targets must contain only 0 and 1")
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    bias_features = np.column_stack(
        [np.ones(features.shape[0], dtype=np.float64), features]
    )
    weights = np.zeros(bias_features.shape[1], dtype=np.float64)
    if selected_activation == "step":
        for _ in range(max_iterations):
            changed = False
            for row, target in zip(bias_features, targets):
                prediction = 1.0 if float(np.dot(row, weights)) >= 0.0 else 0.0
                error = target - prediction
                if error != 0.0:
                    weights += learning_rate * error * row
                    changed = True
            if not changed:
                break
        return weights

    for _ in range(max_iterations):
        weighted_sums = bias_features @ weights
        predictions = 1.0 / (
            1.0 + np.exp(-np.clip(weighted_sums, -500.0, 500.0))
        )
        gradient = bias_features.T @ (predictions - targets)
        gradient /= targets.size
        weights -= learning_rate * gradient
    return weights


def binary_classifier(
    data_set: object,
    weights: FloatArray | list[float],
    activation: str = "sigmoid",
) -> FloatArray:
    """Classify rows of a binary-classification data set.

    Precondition: data_set contains a feature array and weights has one bias
    followed by one weight per feature.
    Postcondition: returns one binary prediction per feature row.
    Effect: no input data or external state is mutated.
    """
    selected_activation = _validate_activation(activation)
    features, _ = _extract_data_set(data_set)
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if weight_array.size not in (features.shape[1], features.shape[1] + 1):
        raise ValueError("weights must include weights for every feature")
    if not np.isfinite(weight_array).all():
        raise ValueError("weights must contain only finite numbers")

    if weight_array.size == features.shape[1]:
        weighted_sums = features @ weight_array
    else:
        weighted_sums = weight_array[0] + features @ weight_array[1:]

    if selected_activation == "sigmoid":
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(weighted_sums, -500.0, 500.0))
        )
        return (probabilities >= 0.5).astype(np.float64)
    return (weighted_sums >= 0.0).astype(np.float64)


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
