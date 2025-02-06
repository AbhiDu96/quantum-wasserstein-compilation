# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

######################################################################################################
# This code is a modified version of qiskit_algorithms.gradients.base.base_estimator_gradient        #
# The main modifications include:                                                                    #
# - An alternated input structure for run()                                                          #
# - Compatibility changed to align with ..reverse_custom.reverse_gradient_custom                     #
######################################################################################################

"""
Abstract base class of custom gradient class.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from copy import copy
import numpy as np
import qiskit_algorithms.gradients.utils

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.transpiler.passes import TranslateParameterizedGates

from qiskit_algorithms.gradients.utils import (
    DerivativeType,
    _assign_unique_parameters,
    _make_gradient_parameters,
    _make_gradient_parameter_values
)

from qiskit_algorithms.algorithm_job import AlgorithmJob


class BaseEstimatorGradientCustom(ABC):
    """Base class for an ``EstimatorGradientCustom`` to compute the gradients of the expectation value."""

    def __init__(
        self,
        estimator: BaseEstimator,
        supported_gates: Sequence[str] = None,
        options: Options | None = None,
        derivative_type: DerivativeType = DerivativeType.REAL,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``.

                    - ``DerivativeType.REAL`` computes :math:`2 \mathrm{Re}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.IMAG`` computes :math:`2 \mathrm{Im}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.COMPLEX`` computes :math:`2 ⟨ψ(ω)|O(θ)|dω ψ(ω)〉`.

                Defaults to ``DerivativeType.REAL``, as this yields e.g. the commonly-used energy
                gradient and this type is the only supported type for function-level schemes like
                finite difference.
        """
        self._estimator: BaseEstimator = estimator
        self.supported_gates = supported_gates
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        self._derivative_type = derivative_type

    @property
    def derivative_type(self) -> DerivativeType:
        """Return the derivative type (real, imaginary or complex).

        Returns:
            The derivative type.
        """
        return self._derivative_type

    def run(
        self,
        circuits: Sequence[QuantumCircuit] | QuantumCircuit,
        observables: Sequence[BaseOperator] | BaseOperator,
        parameter_values: Sequence[float],
        **options,
    ) -> AlgorithmJob:
        """Run the job of the estimator gradient on the given circuits.

        Args:
            circuits: Sequence of quantum circuits
            observables: The (list of) observable(s).
            parameter_values: The list (of list) of parameter values to be bound to the circuit.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting

        Returns:
            The job object of the gradients of the expectation values. The [i, j, k]-th result corresponds to
            the gradient of the i-th circuit `circuits[i]` w.r.t. the k-th parameter ``parameters[k]``,
            for the j-th observable ``observables[j]``
        Raises:
            ValueError: Invalid arguments are given.
        """

        if isinstance(circuits, QuantumCircuit):
            # Allow a single circuit to be passed in.
            circuits = (circuits,)

        for i, circuit in enumerate(circuits):
            if not circuit.num_parameters:
                raise ValueError(f"The {i}-th circuit is not parameterised.")

        gradient_circuits = self._preprocess_circuit(circuits, self.supported_gates)

        if isinstance(observables, BaseOperator):
            # Allow a single observable to be passed in.
            observables = (observables,)

        # Validate the arguments.
        self._validate_arguments(circuits, observables, parameter_values)

        # The priority of run option is as follows:
        # options in ``run`` method > gradient's default options > primitive's default setting.
        opts = copy(self._default_options)
        opts.update_options(**options)

        # Run the job.
        job = AlgorithmJob(
            self._run, circuits, gradient_circuits, observables, parameter_values, **opts.__dict__
        )
        job.submit()
        return job

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        gradient_circuits: Sequence[qiskit_algorithms.gradients.utils.GradientCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[float],
        **options,
    ) -> Sequence[Sequence[Sequence[float]]]:
        """Compute the estimator gradients on the given circuits."""
        raise NotImplementedError()

    @staticmethod
    def _preprocess_circuit(circuits: Sequence[QuantumCircuit], supported_gates: Sequence[str]):
        translator = TranslateParameterizedGates(supported_gates)
        gradient_circuits = []
        for circuit in circuits:
            unrolled = translator(circuit)
            gradient_circuit = _assign_unique_parameters(unrolled)
            gradient_circuits.append(gradient_circuit)
        return gradient_circuits

    @staticmethod
    def _preprocess(
        circuits: Sequence[QuantumCircuit],
        gradient_circuits: Sequence[qiskit_algorithms.gradients.utils.GradientCircuit],
        parameter_values: Sequence[float],
    ) -> tuple[Sequence[float], Sequence[Sequence[Parameter]]]:
        """Preprocess the gradient. This makes a gradient circuit for each circuit. The gradient
        circuit is a transpiled circuit by using the supported gates, and has unique parameters.
        ``parameter_values`` and ``parameters`` are also updated to match the gradient circuit.

        Args:
            circuits: Sequence of QuantumCircuits
            gradient_circuits: Sequence of GradientCircuits
            parameter_values: The list of parameter values to be bound to the circuit.

        Returns:
            The list of gradient circuits, the list of parameter values, and the list of parameters.
            parameter_values and parameters are updated to match the gradient circuit.
        """

        g_parameter_values = []
        for circuit, gradient_circuit in zip(circuits, gradient_circuits):
            g_parameter_values.append(
                _make_gradient_parameter_values(circuit, gradient_circuit, parameter_values)
            )
        g_parameters = [_make_gradient_parameters(gradient_circuit, circuit.parameters) for gradient_circuit, circuit in zip(gradient_circuits, circuits)]
        return g_parameter_values, g_parameters

    def _postprocess(
        self,
        circuits: Sequence[QuantumCircuit],
        gradient_circuits: Sequence[qiskit_algorithms.gradients.utils.GradientCircuit],
        results: Sequence[Sequence[Sequence[float]]],
        parameter_values: Sequence[float],
    ) -> Sequence[Sequence[Sequence[float]]]:
        """Postprocess the gradients. This method computes the gradient of the original circuits
        by applying the chain rule to the gradient of the circuits with unique parameters.

        Args:
            circuits: Sequence of QuantumCircuits
            gradient_circuits: Sequence of GradientCircuits
            results: The computed gradients for the circuits with unique parameters.
            parameter_values: The list of parameter values to be bound to the circuits.

        Returns:
            The gradients of the original circuits.
        """
        gradients = []
        for idx, (circuit, gradient_circuit) in enumerate(zip(circuits, gradient_circuits)):
            parameters = circuit.parameters
            g_parameters = _make_gradient_parameters(gradient_circuit, parameters)
            g_parameter_indices = {param: i for i, param in enumerate(g_parameters)}
            num_observables = len(results[0])
            gradient = np.zeros((num_observables, len(parameters)))
            if self.derivative_type == DerivativeType.COMPLEX:
                # If the derivative type is complex, cast the gradient to complex.
                gradient = gradient.astype("complex")
            for i, parameter in enumerate(parameters):
                for g_parameter, coeff in gradient_circuit.parameter_map[parameter]:
                    # Compute the coefficient
                    if isinstance(coeff, ParameterExpression):
                        local_map = {
                            p: parameter_values[circuit.parameters.data.index(p)]
                            for p in coeff.parameters
                        }
                        bound_coeff = coeff.bind(local_map)
                    else:
                        bound_coeff = coeff
                    # The original gradient is a sum of the gradients of the parameters in the
                    # gradient circuit multiplied by the coefficients.
                    for o in range(num_observables):
                        gradient[o][i] += (
                            float(bound_coeff)
                            * results[idx][o][g_parameter_indices[g_parameter]]
                        )
            gradients.append(gradient)
        return gradients

    @staticmethod
    def _validate_arguments(
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[float],
    ) -> None:
        """Validate the arguments of the ``run`` method.

        Args:
            circuits: The quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.

        Raises:
            ValueError: Invalid arguments are given.
        """

        for i, circuit in enumerate(circuits):
            if len(parameter_values) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_values)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )

        for i, observable_ in enumerate(observables):
            for j, circuit in enumerate(circuits):
                if circuit.num_qubits != observable_.num_qubits:
                    raise ValueError(
                        f"The number of qubits of the {j}-th circuit ({circuit.num_qubits}) does "
                        f"not match the number of qubits of the {i}-th observable "
                        f"({observable_.num_qubits})."
                    )

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and gradient default options,
        where, if the same field is set in both, the gradient's default options override
        the primitive's default setting.

        Returns:
            The gradient default + estimator options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the gradient's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the gradient default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.

        Args:
            options: The fields to update the options

        Returns:
            The gradient default + estimator + run options.
        """
        opts = copy(self._estimator.options)
        opts.update_options(**options)
        return opts
