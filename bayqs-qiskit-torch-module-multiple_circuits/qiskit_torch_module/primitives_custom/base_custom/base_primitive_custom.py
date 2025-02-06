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

############################################################################
# This code is a modified version of qiskit.primitives.base.base_primitive #
# The main modifications include:                                          #
# - Compatibility changed to align with .base_estimator_custom             #
############################################################################

"""PrimitiveCustom abstract base class."""

from abc import ABC
from collections.abc import Sequence

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.providers import Options


class BasePrimitiveCustom(ABC):
    """Primitive abstract base class."""

    def __init__(self, options: dict | None = None):
        self._run_options = Options()
        if options is not None:
            self._run_options.update_options(**options)

    @property
    def options(self) -> Options:
        """Return options values for the estimator.

        Returns:
            options
        """
        return self._run_options

    def set_options(self, **fields):
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._run_options.update_options(**fields)

    @staticmethod
    def _cross_validate_circuit_parameter_values(
        circuits: Sequence[QuantumCircuit], parameter_values: Sequence[float]
    ) -> None:
        for i, circuit in enumerate(circuits):
            if len(parameter_values) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_values)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )


def _isint(obj: Sequence[Sequence[float]] | Sequence[float] | float) -> bool:
    """Check if object is int."""
    int_types = (int, np.integer)
    return isinstance(obj, int_types) and not isinstance(obj, bool)


def _isreal(obj: Sequence[Sequence[float]] | Sequence[float] | float) -> bool:
    """Check if object is a real number: int or float except ``±Inf`` and ``NaN``."""
    float_types = (float, np.floating)
    return _isint(obj) or isinstance(obj, float_types) and float("-Inf") < obj < float("Inf")
