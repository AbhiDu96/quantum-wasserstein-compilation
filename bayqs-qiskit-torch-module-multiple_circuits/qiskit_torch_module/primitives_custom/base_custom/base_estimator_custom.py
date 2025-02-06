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

######################################################################################################
# This code is a modified version of qiskit.primitives.base.base_estimator                           #
# The main modifications include:                                                                    #
# - An alternated input structure for run()                                                          #
# - Compatibility changed to align with ..estimator_custom                                           #
######################################################################################################

r"""

.. estimator-desc:

=====================
Overview of EstimatorCustom
=====================

Estimator class estimates expectation values of quantum circuits and observables.

An estimator is initialized with optional settings. The estimator is used to
create a :class:`~qiskit.providers.JobV1`, via the
:meth:`primitives_custom.EstimatorCustom.run()` method. This method is called
with the following parameters

* circuits: (:math:`psi_i`): a list of quantum circuits to evaluate

* observables (:math:`H_j`): a list of :class:`~qiskit.quantum_info.BaseOperator`
  objects. All observable(s) will be evaluated for all parameter sets in ``parameter_values``

* parameter values (:math:`\theta`): list of values to be bound to the parameters of the quantum circuits
  (Set of floats). All circuits must contain the same number of parameters.

The method returns a :class:`~qiskit.providers.JobV1` object, calling
:meth:`qiskit.providers.JobV1.result()` yields the list of expectation values
plus optional metadata like confidence intervals for the estimation.

.. math::

    \langle\psi_i(\theta)|H_j|\psi_i(\theta)\rangle

Here is an example of how the estimator is used.

.. code-block:: python

    from primitives_custom import EstimatorCustom
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp

    psi1 = RealAmplitudes(num_qubits=2, reps=2)
    psi2 = RealAmplitudes(num_qubits=2, reps=2)
    psi3 = RealAmplitudes(num_qubits=2, reps=2)

    H1 = SparsePauliOp.from_list([("IZ", 1)])
    H2 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

    theta = [0, 1, 2, 3, 4, 5]

    estimator = Estimator()

    # calculate [ [ <psi1(theta)|H1|psi1(theta)> ] ]
    job = estimator.run(psi1, H1, theta)
    job_result = job.result() # It will block until the job finishes.
    print(f"The primitive-job finished with result {job_result}"))

    # calculate [ [ <psi1(theta)|H1|psi1(theta)>,
    #               <psi1(theta)|H2|psi1(theta)> ],
    #             [ <psi2(theta)|H1|psi2(theta)>,
    #               <psi2(theta)|H2|psi2(theta)> ],
    #             [ <psi3(theta)|H1|psi3(theta)>,
    #               <psi3(theta)|H2|psi3(theta)> ] ]
    job2 = estimator.run([psi1, psi2, psi3], [H1, H2], theta)
    job_result = job2.result()
    print(f"The primitive-job finished with result {job_result}")
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from copy import copy

from qiskit.circuit import QuantumCircuit
from qiskit.providers import JobV1 as Job
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_torch_module.primitives_custom.base_custom.base_primitive_custom import BasePrimitiveCustom

T = TypeVar("T", bound=Job)


class BaseEstimatorCustom(BasePrimitiveCustom, Generic[T]):
    """EstimatorCustom base class.

    Base class for EstimatorCustom that estimates expectation values of quantum circuits and observables.
    """

    __hash__ = None

    def __init__(
        self,
        *,
        options: dict | None = None,
    ):
        """
        Creating an instance of an EstimatorCustom.

        Args:
            options: Default options.
        """
        super().__init__(options)

    def run(
        self,
        circuits: Sequence[QuantumCircuit] | QuantumCircuit,
        observables: Sequence[BaseOperator] | BaseOperator,
        parameter_values: Sequence[float],
        **run_options,
    ) -> T:
        """Run the job of the estimation of expectation value(s).

        ``circuits``, and ``observables`` are used to evaluate all pairwise combinations.
        The [i, j]-th element of the result is the expectation of circuit `i` w.r.t. observable `j`

        Args:
            circuits: Sequence of circuits
            observables: one or more observable objects.
            parameter_values: set of concrete parameters to be bound.
            run_options: runtime options used for circuit execution.

        Returns:
            The job object of EstimatorResult.

        Raises:
            TypeError: Invalid argument type given.
            ValueError: Invalid argument values given.
        """

        if isinstance(circuits, QuantumCircuit):
            # Allow a single circuit to be passed in.
            circuits = (circuits,)

        if isinstance(observables, BaseOperator):
            # Allow a single observable to be passed in.
            observables = (observables,)

        # Cross-validation
        self._cross_validate_circuit_parameter_values(circuits, parameter_values)
        self._cross_validate_circuit_observables(circuits, observables)

        # Options
        run_opts = copy(self.options)
        run_opts.update_options(**run_options)

        return self._run(
            circuits,
            observables,
            parameter_values,
            **run_opts.__dict__,
        )

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[float],
        **run_options,
    ) -> T:
        raise NotImplementedError("The subclass of BaseEstimator must implement `_run` method.")

    @staticmethod
    def _cross_validate_circuit_observables(
        circuits: Sequence[QuantumCircuit], observables: Sequence[BaseOperator]
    ) -> None:
        for i, observable in enumerate(observables):
            for j, circuit in enumerate(circuits):
                if circuit.num_qubits != observable.num_qubits:
                    raise ValueError(
                        f"The number of qubits of the {j}-th circuit ({circuit.num_qubits}) does "
                        f"not match the number of qubits of the {i}-th observable "
                        f"({observable.num_qubits})."
                    )
