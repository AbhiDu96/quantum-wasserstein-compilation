# This code is part of Qiskit-Torch-Module[sequential].
# Author: Nico Meyer (nico.meyer@iis.fraunhofer.de)
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator

from collections.abc import Sequence

from qiskit_torch_module.primitives_custom import EstimatorCustom
from qiskit_torch_module.gradients_custom import ReverseEstimatorGradientCustom


class QNN:
    """ This class implements a quantum neural network, which combines forward and backward class for a given setup.

        Args:
            observables: Observables to evaluate, corresponds to output of QNN (default: Pauli-Z on all qubits)
    """

    def __init__(
            self,
            observables: Sequence[BaseOperator] | BaseOperator,
    ):

        # self._circuit, self._encoding_params, self._variational_params =\
        #     generate_alphabetically_ordered_circuit(circuit, encoding_params, variational_params)

        # Estimators for estimation of expectation values and gradients
        self._estimator_expval = EstimatorCustom()
        self._estimator_gradient = ReverseEstimatorGradientCustom()

        # singleton observables:
        if isinstance(observables, BaseOperator):
            self._observables = (observables, )
        else:
            self._observables = observables

    def forward(
            self,
            circuits: Sequence[QuantumCircuit],
            variational_weights: Sequence[float],
    ) -> np.ndarray[np.ndarray[float]]:
        """ Realizes forward pass of QNN, i.e. computation of expectation values

            Args:
                circuits: Sequence of quantum circuits
                variational_weights: Values for trainable weights of the QNN

            Returns:
                Result of forward pass (i.e. expectation values)
        """

        self._validate_weights(circuits, variational_weights)

        job = self._estimator_expval.run(
            circuits=circuits,
            observables=self._observables,
            parameter_values=variational_weights
        )
        return job.result().values

    def backward(
            self,
            circuits: Sequence[QuantumCircuit],
            variational_weights: Sequence[float],
    ) -> np.ndarray[np.ndarray[np.ndarray[float]]]:
        """ Realizes backward pass of QNN, i.e. computation of gradients w.r.t. variational parameters

            Args:
                circuits: Sequence of quantum circuits
                variational_weights: Values for trainable weights of the QNN

            Returns:
                Result of backward pass (i.e. gradients w.r.t. variational parameters)
        """

        self._validate_weights(circuits, variational_weights)

        job = self._estimator_gradient.run(
            circuits=circuits,
            observables=self._observables,
            parameter_values=variational_weights
        )
        return job.result().gradients

    def observables(self) -> Sequence[BaseOperator]:
        """ Return observables
        """
        return self._observables

    def num_observables(self) -> int:
        """ Return number of observables
        """
        return len(self._observables)

    @staticmethod
    def _validate_weights(
            circuits: Sequence[QuantumCircuit],
            weights: Sequence[float],
    ):
        """ Validate parameter values with provided circuits

        Args:
            circuits: Provided quantum circuits
            weights: set of variational weights (one set for all circuits)

        Raises:
            ValueError: Inconsistent number of parameter values
        """
        for i, circuit in enumerate(circuits):
            if len(weights) != circuit.num_parameters:
                raise ValueError(f'The {i}-th circuit contains {circuit.num_parameters} parameters, but {len(weights)} values were provided.')
