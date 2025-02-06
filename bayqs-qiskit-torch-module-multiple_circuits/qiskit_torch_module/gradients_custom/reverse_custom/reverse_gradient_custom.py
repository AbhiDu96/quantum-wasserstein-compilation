# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#########################################################################################
# This code is a modified version of qiskit_algorithms.gradients.reverse_gradient       #
# The main modifications include:                                                       #
# - An alternated input structure for _run()                                            #
# - Efficient gradient computation for multiple (not necessarily commuting) observables #
#########################################################################################

"""Estimator gradients with the classically efficient reverse mode."""

from collections.abc import Sequence
import logging

import qiskit_algorithms.gradients.utils
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector
from qiskit.primitives import Estimator

from qiskit_algorithms.gradients.reverse.bind import bind
from qiskit_algorithms.gradients.reverse.derive_circuit import derive_circuit
from qiskit_algorithms.gradients.reverse.split_circuits import split

from qiskit_torch_module.gradients_custom import BaseEstimatorGradientCustom
from qiskit_torch_module.gradients_custom import EstimatorGradientResultCustom
from qiskit_algorithms.gradients.utils import DerivativeType


logger = logging.getLogger(__name__)


class ReverseEstimatorGradientCustom(BaseEstimatorGradientCustom):
    """Estimator gradients with the classically efficient reverse mode.

    .. note::

        This gradient implementation is based on statevector manipulations and scales
        exponentially with the number of qubits. However, for small system sizes it can be very fast
        compared to circuit-based gradients.

    This class implements the calculation of the expectation gradient as described in
    [1]. By keeping track of two statevectors and iteratively sweeping through each parameterized
    gate, this method scales only linearly with the number of parameters.

    **References:**

        [1]: Jones, T. and Gacon, J. "Efficient calculation of gradients in classical simulations
             of variational quantum algorithms" (2020).
             `arXiv:2009.02823 <https://arxiv.org/abs/2009.02823>`_.

    """

    def __init__(self, derivative_type: DerivativeType = DerivativeType.REAL):
        """
        Args:
            derivative_type: Defines whether the real, imaginary or real plus imaginary part
                of the gradient is returned.
        """
        SUPPORTED_GATES = ["rx", "ry", "rz", "cp", "crx", "cry", "crz"]
        dummy_estimator = Estimator()  # this is required by the base class, but not used
        super().__init__(dummy_estimator, supported_gates=SUPPORTED_GATES, derivative_type=derivative_type)

    @BaseEstimatorGradientCustom.derivative_type.setter
    def derivative_type(self, derivative_type: DerivativeType) -> None:
        """Set the derivative type."""
        self._derivative_type = derivative_type

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        gradient_circuits: Sequence[qiskit_algorithms.gradients.utils.GradientCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[float],
        **options,
    ) -> EstimatorGradientResultCustom:
        """Compute the gradients of the expectation values by the adjoint method."""

        # the metadata only contains the parameters as there are no run configs here
        metadata = (
            {
                "derivative_type": self.derivative_type,
            }
        )

        g_parameter_values, g_parameters = self._preprocess(
            circuits, gradient_circuits, parameter_values
        )

        gradients = []
        for gradient_circuit_, parameter_values_, parameters_ in zip(gradient_circuits, g_parameter_values, g_parameters):
            # temporary variables for easier access
            circuit = gradient_circuit_.gradient_circuit

            # split the circuit and generate lists of unitaries [U_1, U_2, ...] and
            # parameters [p_1, p_2, ...] in these unitaries
            unitaries, paramlist = split(circuit, parameters=parameters_)

            parameter_binds = dict(zip(circuit.parameters, parameter_values_))
            bound_circuit = bind(circuit, parameter_binds)

            # initialize state variables -- we use the same naming as in the paper
            phi = Statevector(bound_circuit)
            lams = [_evolve_by_operator(observable, phi) for observable in observables]

            # store gradients in a dictionary to return them in the correct order
            # important to do full dictionary generation for each observable instance to prevent call-by-reference
            gradss = [{param: 0j for param in parameters_} for _ in observables]

            num_parameters = len(unitaries)
            for j in reversed(range(num_parameters)):
                unitary_j = unitaries[j]

                # We currently only support gates with a single parameter -- which is reflected
                # in self.SUPPORTED_GATES -- but generally we could also support gates with multiple
                # parameters per gate
                parameter_j = paramlist[j][0]

                # get the analytic gradient d U_j / d p_j and bind the gate
                deriv = derive_circuit(unitary_j, parameter_j)
                for _, gate in deriv:
                    bind(gate, parameter_binds, inplace=True)

                # iterate the state variable
                unitary_j_dagger = bind(unitary_j, parameter_binds).inverse()
                phi = phi.evolve(unitary_j_dagger)

                # pre-compute here, as it is the same for all observables
                phis_evolved = [phi.evolve(gate).data for _, gate in deriv]
                grad = [
                    sum(
                        coeff * lam.conjugate().data.dot(phi_evolved) for (coeff, _), phi_evolved in
                        zip(deriv, phis_evolved)
                    )
                    for lam in lams
                ]

                # Compute the full gradient (real and complex parts) as all information is available.
                # Later, based on the derivative type, cast to real/imag/complex.
                for o, g in enumerate(grad):
                    gradss[o][parameter_j] += g

                if j > 0:
                    lams = [lam.evolve(unitary_j_dagger) for lam in lams]

            gradient = [np.array(list(grads.values())) for grads in gradss]
            gradients.append(self._to_derivtype(gradient))
        result_final = self._postprocess(circuits, gradient_circuits, gradients, parameter_values)
        return EstimatorGradientResultCustom(np.array(result_final, dtype=np.float32), metadata)

    def _to_derivtype(self, gradient):
        # this disable is needed as Pylint does not understand derivative_type is a property if
        # it is only defined in the base class and the getter is in the child
        # pylint: disable=comparison-with-callable
        if self.derivative_type == DerivativeType.REAL:
            return 2 * np.real(gradient)
        if self.derivative_type == DerivativeType.IMAG:
            return 2 * np.imag(gradient)

        return 2 * gradient


def _evolve_by_operator(operator, state):
    """Evolve the Statevector state by operator."""
    # try casting to sparse matrix and use sparse matrix-vector multiplication, which is
    # a lot faster than using Statevector.evolve
    try:
        spmatrix = operator.to_matrix(sparse=True)
        evolved = spmatrix @ state.data
        return Statevector(evolved)
    except (TypeError, AttributeError):
        logger.info("Operator is not castable to a sparse matrix, using Statevector.evolve.")
    return state.evolve(operator)
