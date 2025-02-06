# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#########################################################################################
# This code is a modified version of qiskit.primitives.estimator                        #
# The main modifications include:                                                       #
# - An alternated input structure for _run()                                            #
# - Efficient gradient computation for multiple (not necessarily commuting) observables #
#########################################################################################

"""
EstimatorCustom class
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import bound_circuit_to_instruction

from qiskit_torch_module.primitives_custom import BaseEstimatorCustom, EstimatorResultCustom


class EstimatorCustom(BaseEstimatorCustom[PrimitiveJob[EstimatorResultCustom]]):
    """
    Reference implementation of :class:`BaseEstimator`.

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the exact expectation
          values. Otherwise, it samples from normal distributions with standard errors as standard
          deviations using normal distribution approximation.

        - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the normal distribution. If shots is None,
          this option is ignored.
    """

    def __init__(self, *, options: dict | None = None):
        """
        Args:
            options: Default options.

        Raises:
            QiskitError: if some classical bits are not used for measurements.
        """
        super().__init__(options=options)

    def _call(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[float],
        shots: int | None = None,
        seed: int | None = None,
        **run_options,
    ) -> EstimatorResultCustom:
        if seed is None:
            rng = np.random.default_rng()
        elif isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # Initialize metadata
        metadata: list[list[dict[str, Any]]] = [[{} for _ in range(len(observables))] for _ in range(len(circuits))]
        expectation_values = []
        for circuit, metadatum in zip(circuits, metadata):
            # bind parameters to circuit
            # CAUTION: Qiskit always does this in alphabetical order!
            bound_circuit = circuit if 0 == len(circuit.parameters) else circuit.assign_parameters(dict(zip(circuit.parameters, parameter_values)))

            final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
            expectation_values_ = [final_state.expectation_value(obs) for obs in observables]

            if shots is None:
                expectation_values.append(expectation_values_)
            else:
                expectation_values__ = []
                for obs_, expectation_value_, metadatum_ in zip(observables, expectation_values_, metadatum):
                    expectation_value = np.real_if_close(expectation_value_)
                    sq_obs = (obs_ @ obs_).simplify(atol=0)
                    sq_exp_val = np.real_if_close(final_state.expectation_value(sq_obs))
                    variance = sq_exp_val - expectation_value ** 2
                    variance = (max(variance, 0)).astype(np.float32)
                    standard_error = np.sqrt(variance / shots)
                    expectation_value_with_error = rng.normal(expectation_value, standard_error)
                    expectation_values__.append(expectation_value_with_error)
                    metadatum_["shots"] = shots
                    metadatum_["variance"] = variance
                expectation_values.append(expectation_values__)

        return EstimatorResultCustom(np.real_if_close(np.array(expectation_values)), metadata)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[float],
        **run_options,
    ):

        job = PrimitiveJob(
            self._call, circuits, observables, parameter_values,
            run_options.pop("shots", None), run_options.pop("seed", None)
        )
        job._submit()
        return job
